import os
import random as _random

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from monai.data import CacheDataset, Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    MapTransform,
    Orientationd,
    NormalizeIntensityd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    SpatialPadd,
    EnsureTyped,
)


def build_data_list(data_root, split_dir, modalities, has_labels=True):
    """Build a list of dicts for MONAI datasets.

    Each dict has:
        "image": list of 4 modality paths
        "label": path to seg file (only if has_labels)
        "case_id": str

    split_dir may be an absolute path (e.g. synthetic_data dir); if so,
    data_root is ignored for path construction.
    """
    split_path = split_dir if os.path.isabs(split_dir) else os.path.join(data_root, split_dir)
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


class ExtractRandomSlice2d(MapTransform):
    """
    Extracts a single random non-empty axial slice from a loaded 3-D volume.

    Expects volumes already processed by EnsureChannelFirst + Orientationd(RAS),
    so spatial shape is (C, H, W, D) where D is the axial (S) dimension.
    After extraction each key becomes (C, H, W).
    """

    def __init__(self, keys, label_key: str = "label"):
        super().__init__(keys, allow_missing_keys=False)
        self.label_key = label_key

    def __call__(self, data):
        d = dict(data)
        lbl = d[self.label_key]                        # (1, H, W, D)
        lbl_arr = np.asarray(lbl)
        D = lbl_arr.shape[-1]

        nonempty = [z for z in range(D) if lbl_arr[0, :, :, z].max() > 0]
        z = int(_random.choice(nonempty)) if nonempty else D // 2

        for key in self.key_iterator(d):
            d[key] = d[key][..., z]                    # (C, H, W, D) → (C, H, W)
        return d


def get_train_transforms_2d():
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ExtractRandomSlice2d(keys=["image", "label"]),
        SpatialPadd(keys=["image", "label"], spatial_size=(220, 220)),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        RandGaussianNoised(keys="image", prob=0.2, mean=0.0, std=0.1),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms_2d():
    return Compose([
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ExtractRandomSlice2d(keys=["image", "label"]),
        SpatialPadd(keys=["image", "label"], spatial_size=(220, 220)),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_dataloaders_2d(config, synthetic_data_dir=None):
    """Build 2-D slice train/val dataloaders for the hierarchical model.

    synthetic_data_dir: optional path to GAN-generated cases. Appended to
    training split only.
    """
    data_root = config["data_root"]
    modalities = config["modalities"]
    seed = config["seed"]
    val_frac = config["val_split"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    all_data = build_data_list(data_root, config["train_dir"], modalities, has_labels=True)

    if "data_fraction" in config and config["data_fraction"] < 1.0:
        _random.seed(seed)
        subset_size = int(len(all_data) * config["data_fraction"])
        all_data = _random.sample(all_data, subset_size)

    train_data, val_data = split_train_val(all_data, val_frac, seed)

    if synthetic_data_dir:
        synth_list = build_data_list(
            os.path.abspath(synthetic_data_dir),
            os.path.abspath(synthetic_data_dir),
            modalities,
            has_labels=True,
        )
        train_data = train_data + synth_list
        print(f"Synthetic data: added {len(synth_list)} cases to training split")

    train_ds = Dataset(data=train_data, transform=get_train_transforms_2d())
    val_ds = Dataset(data=val_data, transform=get_val_transforms_2d())

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )
    return train_loader, val_loader


def get_dataloaders(config, synthetic_data_dir=None):
    """Build train and val dataloaders from config.

    synthetic_data_dir: optional path to a directory of GAN-generated cases
    (same layout as the BraTS training dir). Cases there are appended to the
    training split only — validation always uses real data.
    """
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

    if synthetic_data_dir:
        synth_list = build_data_list(
            os.path.abspath(synthetic_data_dir),
            os.path.abspath(synthetic_data_dir),
            modalities,
            has_labels=True,
        )
        train_data = train_data + synth_list
        print(f"Synthetic data: added {len(synth_list)} cases to training split")

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
