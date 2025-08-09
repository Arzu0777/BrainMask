import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def generate_gradcam(model, input_tensor, target_layer, device):
    input_tensor = input_tensor.unsqueeze(0).to(device)
    model.eval()

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_full_backward_hook(backward_hook)

    output, *_ = model(input_tensor)
    class_score = output.sigmoid().mean()
    model.zero_grad()
    class_score.backward()

    grad = gradients[0]
    act = activations[0]

    pooled_grad = grad.mean(dim=(2, 3), keepdim=True)
    cam = (pooled_grad * act).sum(dim=1).squeeze()
    cam = F.relu(cam)

    cam -= cam.min()
    cam /= cam.max() + 1e-8
    cam = cam.detach().cpu().numpy()

    handle_fw.remove()
    handle_bw.remove()

    return cam

def overlay_heatmap_on_image(image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    cam_resized = np.uint8(255 * cam_resized)

    # Apply color map
    heatmap = cv2.applyColorMap(cam_resized, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Normalize image and convert to RGB uint8
    image_norm = image - np.min(image)
    image_norm /= np.max(image_norm) + 1e-8
    image_rgb = np.uint8(255 * np.stack([image_norm] * 3, axis=-1))

    # Overlay with alpha blending
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap, alpha, 0)

    return overlay

def visualize_gradcam_result(image_tensor, mask_tensor, cam, save_path):
    img = image_tensor.detach().cpu().numpy()
    mask = mask_tensor.detach().cpu().numpy()

    # Handle BraTS image with 12 channels (4 modalities x 3 slices)
    if img.shape[0] == 12:
        img = img[4]  # FLAIR, middle slice
    elif img.shape[0] > 1:
        img = img.mean(0)
    else:
        img = img[0]

    img -= img.min()
    img /= img.max() + 1e-8

    if mask.ndim == 3:
        mask = mask[0]
    mask -= mask.min()
    mask /= mask.max() + 1e-8

    overlay = overlay_heatmap_on_image(img, cam, alpha=0.5)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("Original Image")
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Segmented Mask")
    axs[2].imshow(overlay)
    axs[2].set_title("Grad-CAM Overlay")

    for ax in axs:
        ax.axis('off')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
