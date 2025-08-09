import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from scripts.eval_utils import (
    dice_score as eval_dice_score,
    iou_score as eval_iou_score,
    recall_score as eval_recall_score,
    hausdorff_distance95 as eval_hd95
)
from scripts.visualize import plot_sample

class SoftDiceLoss(nn.Module):  
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = torch.sigmoid(logits)
        intersection = (logits * targets).sum(dim=(1, 2, 3))
        union = logits.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-4, device=None, checkpoint_dir="checkpoints"):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    bce = nn.BCEWithLogitsLoss()
    dice = SoftDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    scaler = torch.amp.GradScaler('cuda')

    best_dice = 0.0
    train_losses, val_dices, val_ious, val_recalls, val_hd95s = [], [], [], [], []

    os.makedirs("results", exist_ok=True) 
    os.makedirs("results/epoch_samples", exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    csv_path = os.path.join("results", "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Dice", "Val IoU", "Val Recall", "Val HD95", "LR"])

    patience = 10
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            if not torch.isfinite(images).all() or not torch.isfinite(masks).all():
                raise ValueError("NaNs detected in input batch")

            with torch.amp.autocast(device_type=device.type):
                outputs, ds2, ds3, ds4 = model(images)

                if not torch.isfinite(outputs).all():
                    raise ValueError("NaNs detected in model output")

                loss = (
                    0.5 * bce(outputs, masks) + 0.5 * dice(outputs, masks) +
                    0.2 * (bce(ds2, masks) + dice(ds2, masks)) +
                    0.2 * (bce(ds3, masks) + dice(ds3, masks)) +
                    0.2 * (bce(ds4, masks) + dice(ds4, masks))
                )

                if not torch.isfinite(loss):
                    raise ValueError("NaN loss detected")

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        epoch_dices, epoch_ious, epoch_recalls, epoch_hd95s = [], [], [], []
        visualized = False

        for idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)

            with torch.amp.autocast(device_type=device.type):
                preds_all = []
                for flip in [None, "horizontal"]:
                    aug = images.flip(-1) if flip == "horizontal" else images
                    out, *_ = model(aug)
                    out = out.flip(-1) if flip == "horizontal" else out
                    preds_all.append(torch.sigmoid(out))

                outputs = torch.stack(preds_all).mean(0)
                preds = (outputs > 0.5).float()

            epoch_dices.append(eval_dice_score(preds, masks).item())
            epoch_ious.append(eval_iou_score(preds, masks).item())
            epoch_recalls.append(eval_recall_score(preds, masks).item())
            epoch_hd95s.append(eval_hd95(preds, masks).item())

            if not visualized:
                for i in range(images.shape[0]):
                    mask = masks[i]
                    pred = preds[i]
                    if mask.sum().item() > 10:
                        if mask.dim() == 3 and mask.shape[0] == 1:
                            mask = mask.squeeze(0)
                        if pred.dim() == 3 and pred.shape[0] == 1:
                            pred = pred.squeeze(0)
                        plot_sample(
                            input_tensor=images[i],
                            target_tensor=mask,
                            pred_tensor=pred,
                            save_path="results/epoch_samples",
                            epoch=epoch, 
                            tag="val"
                        )
                        visualized = True
                        break

        avg_dice = sum(epoch_dices) / len(epoch_dices)
        avg_iou = sum(epoch_ious) / len(epoch_ious)
        avg_recall = sum(epoch_recalls) / len(epoch_recalls)
        avg_hd95 = sum(epoch_hd95s) / len(epoch_hd95s)

        val_dices.append(avg_dice)
        val_ious.append(avg_iou)
        val_recalls.append(avg_recall)
        val_hd95s.append(avg_hd95)

        scheduler.step(avg_dice)

        print(f"[{epoch+1}] Loss: {train_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, Recall: {avg_recall:.4f}, HD95: {avg_hd95:.2f}")

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, avg_dice, avg_iou, avg_recall, avg_hd95, optimizer.param_groups[0]["lr"]])

        if avg_dice > best_dice:
            best_dice = avg_dice
            epochs_no_improve = 0
            ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Saved Best Model to {ckpt_path} (Dice: {best_dice:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("⏹️ Early stopping triggered.")
                break

    return train_losses, val_dices, val_ious, val_recalls, val_hd95s

def plot_metrics(train_losses, val_dices, val_ious, val_recalls, val_hd95s):
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label="Train Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(val_dices, label="Val Dice", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Validation Dice")
    plt.grid()
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(val_ious, label="Val IoU", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("IoU Score")
    plt.title("Validation IoU")
    plt.grid()
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(val_recalls, label="Val Recall", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Validation Recall (Sensitivity)")
    plt.grid()
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(val_hd95s, label="Val HD95", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("HD95")
    plt.title("Validation Hausdorff Distance (95%)")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/training_curves.png")
    plt.close()