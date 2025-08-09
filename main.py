import torch
import os
from scripts.model import UNet
from scripts.dataloader import get_loaders
from scripts.train import train_model, plot_metrics
from scripts.visualize import visualize_samples
from scripts.visualize import plot_sample
from scripts.gradcam import generate_gradcam, visualize_gradcam_result

if __name__ == '__main__':
    data_dir = "data/BraTS"
    checkpoint_dir = "checkpoints"
    result_dir = "results/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=12, out_channels=1)

    train_loader, val_loader = get_loaders(data_dir=data_dir, batch_size=2)

    train_losses, val_dices, val_ious, val_recalls, val_hd95s = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=30,
        lr=1e-4,
        device=device,
        checkpoint_dir=checkpoint_dir
    )

    plot_metrics(train_losses, val_dices, val_ious, val_recalls, val_hd95s)

    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    with torch.no_grad():
        for batch in val_loader:
            images, masks = batch
            images, masks = images.to(device), masks.to(device)
            out, _, _, _ = model(images)
            preds = torch.sigmoid(out) > 0.5

            plot_sample(images[1], masks[1], preds[1], save_path=result_dir, tag="best_model")
            break

    visualize_samples(model, val_loader, save_dir=result_dir, device=device)

    print("\n🧠 Running Grad-CAM for one validation sample...")
    model.eval()

    num_samples = 6
    sampled = 0

    for images, masks in val_loader:
        batch_size = images.size(0)

        for i in range(batch_size):
            image = images[i].to(device)
            mask = masks[i].to(device)

            cam = generate_gradcam(model, image, model.target_layer, device)

            save_path = os.path.join(result_dir, f"gradcam_panel_{sampled + 1}.png")
            visualize_gradcam_result(image, mask, cam, save_path=save_path)

            print(f"✅ Saved Grad-CAM panel: gradcam_panel_{sampled + 1}.png")

            sampled += 1
            if sampled >= num_samples:
                break

        if sampled >= num_samples:
            break
