import os
import math
import random

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data import build_data_list


def compute_tumour_centre(label_vol):
    """Find the centre of the tumour region (any nonzero label)."""
    coords = np.argwhere(label_vol > 0)
    if len(coords) == 0:
        return None
    return coords.mean(axis=0).astype(int)


def crop_around_centre(vol, centre, crop_size):
    """Crop a 3D volume around a centre point, padding if needed."""
    d, h, w = vol.shape[-3:]
    cd, ch, cw = crop_size
    hd, hh, hw = cd // 2, ch // 2, cw // 2

    # Clamp centre to ensure valid crop
    ci, cj, ck = centre
    ci = max(hd, min(ci, d - hd))
    cj = max(hh, min(cj, h - hh))
    ck = max(hw, min(ck, w - hw))

    return vol[..., ci - hd:ci + hd, cj - hh:cj + hh, ck - hw:ck + hw]


def create_noisy_input(scan, label, crop_size=(96, 96, 96)):
    """Create the noisy input z as described in the paper.

    1. Normalize scan to [-1, 1]
    2. Replace tumour voxels with Gaussian noise
    3. Neighbouring voxels replaced with decreasing probability (spherical decay)
    4. Renormalize to [-1, 1]

    Args:
        scan: (4, D, H, W) numpy array — 4 modality channels
        label: (D, H, W) numpy array — segmentation labels

    Returns:
        noisy_scan: (4, D, H, W) numpy array
    """
    noisy = scan.copy()

    # Find tumour centre and size for probability decay
    tumour_mask = label > 0
    if tumour_mask.sum() == 0:
        return noisy

    coords = np.argwhere(tumour_mask)
    centre = coords.mean(axis=0)
    max_size = max(
        coords[:, 0].max() - coords[:, 0].min(),
        coords[:, 1].max() - coords[:, 1].min(),
        coords[:, 2].max() - coords[:, 2].min(),
    )

    # Replace tumour voxels with noise
    noise = np.random.randn(*noisy.shape).astype(np.float32)
    for c in range(noisy.shape[0]):
        noisy[c][tumour_mask] = noise[c][tumour_mask]

    # Spherical noise decay for neighbouring voxels (Eq. 1-2 from paper)
    exponent_base = -0.2 / 68 * max_size + 1.1 - 96 * 0.2 / 68
    all_coords = np.argwhere(~tumour_mask)
    if len(all_coords) > 0:
        distances = np.linalg.norm(all_coords - centre, axis=1)
        probs = 83.0 / (np.power(exponent_base, distances) + 82.0)
        rand_vals = np.random.rand(len(all_coords))
        replace_mask = rand_vals < probs
        replace_coords = all_coords[replace_mask]
        for c in range(noisy.shape[0]):
            noisy[c][replace_coords[:, 0], replace_coords[:, 1], replace_coords[:, 2]] = \
                noise[c][replace_coords[:, 0], replace_coords[:, 1], replace_coords[:, 2]]

    # Renormalize to [-1, 1]
    for c in range(noisy.shape[0]):
        vmin, vmax = noisy[c].min(), noisy[c].max()
        if vmax - vmin > 0:
            noisy[c] = 2.0 * (noisy[c] - vmin) / (vmax - vmin) - 1.0

    return noisy


class GanTrainDataset(Dataset):
    """Dataset for GliGAN training.

    For each case: loads 4 modalities + seg label, crops 96³ around tumour,
    normalizes, and creates noisy input z.

    Returns dict with:
        "noisy_input": (5, 96, 96, 96) — 4 noisy modalities + 1 label channel
        "real_scan": (4, 96, 96, 96) — clean 4-modality scan
        "label": (1, 96, 96, 96) — segmentation label
    """

    def __init__(self, data_root, train_dir, modalities, crop_size=(96, 96, 96), data_fraction=1.0, seed=42):
        self.crop_size = crop_size
        self.data_list = build_data_list(data_root, train_dir, modalities, has_labels=True)

        if data_fraction < 1.0:
            random.seed(seed)
            n = int(len(self.data_list) * data_fraction)
            self.data_list = random.sample(self.data_list, n)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        entry = self.data_list[idx]

        # Load all 4 modalities
        modality_vols = []
        for path in entry["image"]:
            vol = nib.load(path).get_fdata().astype(np.float32)
            modality_vols.append(vol)
        scan = np.stack(modality_vols, axis=0)  # (4, D, H, W)

        # Load label
        label = nib.load(entry["label"]).get_fdata().astype(np.float32)  # (D, H, W)

        # Find tumour centre
        centre = compute_tumour_centre(label)
        if centre is None:
            # No tumour — use volume centre
            centre = np.array([s // 2 for s in label.shape])

        # Crop around tumour centre
        scan = crop_around_centre(scan, centre, self.crop_size)
        label = crop_around_centre(label, centre, self.crop_size)

        # Normalize scan to [-1, 1]
        for c in range(scan.shape[0]):
            vmin, vmax = scan[c].min(), scan[c].max()
            if vmax - vmin > 0:
                scan[c] = 2.0 * (scan[c] - vmin) / (vmax - vmin) - 1.0

        # Create noisy input
        noisy_scan = create_noisy_input(scan, label, self.crop_size)

        # Build tensors
        label_ch = label[np.newaxis, ...]  # (1, D, H, W)
        # Normalize label to [0, 1] range for network input
        label_norm = label_ch / max(label_ch.max(), 1.0)

        noisy_input = np.concatenate([noisy_scan, label_norm], axis=0)  # (5, D, H, W)

        return {
            "noisy_input": torch.from_numpy(noisy_input),
            "real_scan": torch.from_numpy(scan),
            "label": torch.from_numpy(label_ch),
        }
