import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import nibabel as nib
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import warnings

class BraTSDataset(Dataset):
    def __init__(self, data_dir, transform=None, slice_range=1, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.slice_range = slice_range
        self.mode = mode.lower()

        self.patients = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if len(self.patients) == 0:
            raise ValueError(f"No valid patients found in {data_dir}")

    def _load_nifti(self, path):
        return nib.load(path).get_fdata()

    def _get_path(self, patient, modality):
        for ext in [".nii", ".nii.gz"]:
            path = os.path.join(self.data_dir, patient, f"{patient}_{modality}{ext}")
            if os.path.exists(path):
                return path
        warnings.warn(f"Missing file: {patient}_{modality}")
        return None

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient = self.patients[idx]
        modalities = ["flair", "t1", "t1ce", "t2"]
        image_stack = []

        for mod in modalities:
            path = self._get_path(patient, mod)
            if path is None:
                raise FileNotFoundError(f"{mod} modality missing for patient {patient}")
            img = self._load_nifti(path)
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            image_stack.append(img)

        mask_vol = self._load_nifti(self._get_path(patient, "seg"))
        mask_vol = (mask_vol > 0).astype(np.float32)

        # Try to find a slice with non-empty mask (max 5 tries)
        for _ in range(5):
            if self.mode == 'train':
                z = random.randint(self.slice_range, mask_vol.shape[2] - self.slice_range - 1)
            else:
                z = mask_vol.shape[2] // 2
            mask = mask_vol[:, :, z]
            if mask.sum() > 0 or self.mode != 'train':
                break
        else:
            # Fallback: take the center slice
            z = mask_vol.shape[2] // 2
            mask = mask_vol[:, :, z]

        image = np.stack([
            mod[:, :, z + i]
            for i in range(-self.slice_range, self.slice_range + 1)
            for mod in image_stack
        ], axis=0)

        mask = mask[None, ...]

        if self.transform:
            augmented = self.transform(image=image.transpose(1, 2, 0), mask=mask[0])
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)

        return image, mask


def get_loaders(data_dir, batch_size=4, num_workers=0, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_transform = A.Compose([
        A.Resize(240, 240),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ElasticTransform(alpha=1.0, sigma=50.0, p=0.2),
        A.GridDistortion(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.Affine(translate_percent=0.0625, scale=1.1, rotate=15, p=0.5),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=0, std=1),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(240, 240),
        A.Normalize(mean=0, std=1),
        ToTensorV2()
    ])

    full_dataset = BraTSDataset(data_dir, transform=None)
    n = len(full_dataset)
    train_size = int(0.8 * n)
    val_size = n - train_size

    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # Assign transforms after split
    train_set.dataset.transform = train_transform
    val_set.dataset.transform = val_transform
    train_set.dataset.mode = 'train'
    val_set.dataset.mode = 'val'

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
