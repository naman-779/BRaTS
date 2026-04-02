import os

import torch
from sklearn.model_selection import train_test_split

from monai.data import CacheDataset, Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    EnsureTyped,
)


def build_data_list(data_root, split_dir, modalities, has_labels=True):
    """Build a list of dicts for MONAI datasets.

    Each dict has:
        "image": list of 4 modality paths
        "label": path to seg file (only if has_labels)
        "case_id": str
    """
    split_path = os.path.join(data_root, split_dir)
    cases = sorted(
        [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    )
    data_list = []
    for case_id in cases:
        case_dir = os.path.join(split_path, case_id)
        image_paths = [os.path.join(case_dir, f"{case_id}-{mod}.nii.gz") for mod in modalities]

        # Verify all modality files exist
        if not all(os.path.exists(p) for p in image_paths):
            continue

        entry = {"image": image_paths, "case_id": case_id}

        if has_labels:
            seg_path = os.path.join(case_dir, f"{case_id}-seg.nii.gz")
            if not os.path.exists(seg_path):
                continue
            entry["label"] = seg_path

        data_list.append(entry)
    return data_list


def split_train_val(data_list, val_frac, seed):
    """Split data_list into train and val subsets."""
    train_data, val_data = train_test_split(
        data_list, test_size=val_frac, random_state=seed
    )
    return train_data, val_data


def get_train_transforms(roi_size):
    return Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # Spatial augmentations
            RandSpatialCropd(
                keys=["image", "label"], roi_size=roi_size, random_size=False
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            # Intensity augmentations (image only)
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.1),
            RandGaussianSmoothd(
                keys="image",
                prob=0.2,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
            ),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_val_transforms():
    return Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_inference_transforms():
    return Compose(
        [
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ]
    )


def get_dataloaders(config):
    """Build train and val dataloaders from config."""
    data_root = config["data_root"]
    modalities = config["modalities"]
    seed = config["seed"]
    val_frac = config["val_split"]
    roi_size = config["training"]["roi_size"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    cache_rate = config["training"]["cache_rate"]

    all_data = build_data_list(data_root, config["train_dir"], modalities, has_labels=True)

    # Use only a fraction of data if specified (for quick baseline)
    if "data_fraction" in config and config["data_fraction"] < 1.0:
        import random
        random.seed(seed)
        subset_size = int(len(all_data) * config["data_fraction"])
        all_data = random.sample(all_data, subset_size)

    train_data, val_data = split_train_val(all_data, val_frac, seed)

    train_ds = CacheDataset(
        data=train_data,
        transform=get_train_transforms(roi_size),
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    val_ds = Dataset(data=val_data, transform=get_val_transforms())

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory
    )
    return train_loader, val_loader
