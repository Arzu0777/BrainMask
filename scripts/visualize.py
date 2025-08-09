import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from scripts.model import UNet
from scripts.dataloader import BraTSDataset
from torch.utils.data import DataLoader
#from pytorch_grad_cam import GradCAM
#from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
#from pytorch_grad_cam.utils.image import show_cam_on_image
from scripts.eval_utils import dice_score, iou_score

sns.set(style="whitegrid")

def normalize_img(img):
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img

def visualize_prediction(model, image, mask, device, save_path=None, add_metrics=True):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        output = torch.sigmoid(model(image)[0])
        pred = output.squeeze().cpu().numpy()

    flair = image.squeeze().cpu().numpy()[0]  # FLAIR assumed at channel 0
    mask = mask.squeeze().cpu().numpy()
    pred_bin = pred > 0.5

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(flair, cmap='gray')
    axes[0].set_title("FLAIR Image")
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_bin, cmap='gray')
    axes[2].set_title("Prediction")

    if add_metrics:
        pred_tensor = torch.tensor(pred_bin).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
        dice = dice_score(pred_tensor, mask_tensor).item()
        iou = iou_score(pred_tensor, mask_tensor).item()
        axes[2].set_title(f"Prediction\nDice: {dice:.3f}, IoU: {iou:.3f}")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📸 Saved prediction figure: {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_samples(model, val_loader, save_dir="results/samples/", device=None, max_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    count = 0
    for images, masks in val_loader:
        images, masks = images.to(device), masks.to(device)
        for i in range(images.size(0)):
            if count >= max_samples:
                return
            image, mask = images[i].cpu(), masks[i].cpu()
            visualize_prediction(model, image, mask, device, save_path=f"{save_dir}/sample_{count}_pred.png")
            #save_cam_visualization(model, image.unsqueeze(0).to(device), mask, save_path=f"{save_dir}/sample_{count}_cam.png", device=device)
            count += 1

def visualize_grid(dataset, model, device, save_path="results/grid.png", indices=[0, 1, 2, 3]):
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    model.to(device).eval()

    for i, idx in enumerate(indices):
        image, mask = dataset[idx]
        image_t = image.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = torch.sigmoid(model(image_t)[0]).squeeze().cpu().numpy()
        flair = image[0].numpy()
        mask = mask.squeeze().numpy()
        pred_bin = pred > 0.5

        axes[i, 0].imshow(flair, cmap="gray")
        axes[i, 0].set_title("FLAIR")
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(pred_bin, cmap="gray")
        axes[i, 2].set_title("Prediction")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🧩 Saved comparison grid: {save_path}")
    plt.close()

def visualize_single(data_dir, model_path, idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BraTSDataset(data_dir)
    image, mask = dataset[idx]

    model = UNet(in_channels=12, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    visualize_prediction(model, image, mask, device)
    #save_cam_visualization(model, image.unsqueeze(0).to(device), mask, save_path=f"results/cam_sample_{idx}.png", device=device)

def plot_sample(input_tensor, target_tensor, pred_tensor, save_path, epoch=None, tag="train"):
    input_tensor = input_tensor.cpu().squeeze().numpy()
    target_tensor = target_tensor.cpu().squeeze().numpy()
    pred_tensor = pred_tensor.cpu().squeeze().numpy()

    # Select a single channel for visualization (e.g., channel 6)
    # You can customize this index based on which slice or modality you prefer
    if input_tensor.ndim == 3 and input_tensor.shape[0] > 1:
        input_vis = input_tensor[6]  # pick channel 6 as example
    else:
        input_vis = input_tensor

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Input (ch 6)", "Ground Truth", "Prediction"]
    images = [input_vis, target_tensor, pred_tensor]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    if epoch is not None:
        filename = f"{tag}_epoch_{epoch}.png"
    else:
        filename = f"{tag}_final.png"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(os.path.join(save_path, filename))
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()

    dataset = BraTSDataset(args.data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=12, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    if args.grid:
        visualize_grid(dataset, model, device, save_path="results/grid_comparison.png", indices=[0, 1, 2, 3])
    else:
        visualize_single(args.data_dir, args.model_path, idx=args.idx)
