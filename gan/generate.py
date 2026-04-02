import argparse
import os
import random

import nibabel as nib
import numpy as np
import torch

from gan.dataset import compute_tumour_centre, create_noisy_input
from src.data import build_data_list
from src.utils import get_device, load_config, set_seed

from monai.networks.nets import SwinUNETR


def find_healthy_location(label_vol, crop_size):
    """Find a random location in the brain that is tumour-free and has enough tissue."""
    d, h, w = label_vol.shape
    cd, ch, cw = crop_size
    hd, hh, hw = cd // 2, ch // 2, cw // 2

    brain_mask = label_vol == 0  # background or healthy tissue
    for _ in range(100):  # try up to 100 random locations
        ci = random.randint(hd, d - hd - 1)
        cj = random.randint(hh, h - hh - 1)
        ck = random.randint(hw, w - hw - 1)

        patch = label_vol[ci - hd:ci + hd, cj - hh:cj + hh, ck - hw:ck + hw]
        if patch.sum() == 0:  # no tumour in this region
            return (ci, cj, ck)
    return None


def extract_tumour_label(label_vol, crop_size):
    """Extract a cropped tumour label from a case."""
    centre = compute_tumour_centre(label_vol)
    if centre is None:
        return None, None

    d, h, w = label_vol.shape
    cd, ch, cw = crop_size
    hd, hh, hw = cd // 2, ch // 2, cw // 2

    ci, cj, ck = centre
    ci = max(hd, min(ci, d - hd))
    cj = max(hh, min(cj, h - hh))
    ck = max(hw, min(ck, w - hw))

    cropped = label_vol[ci - hd:ci + hd, cj - hh:cj + hh, ck - hw:ck + hw]
    return cropped, (ci, cj, ck)


def generate_synthetic_data(config, checkpoint_path, output_dir, max_cases=None):
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    gcfg = config["gan"]["generator"]
    gen_cfg = config["gan"]["generation"]
    crop_size = tuple(gen_cfg["crop_size"])
    num_synthetic = gen_cfg["num_synthetic_per_case"]

    # Build and load generator
    generator = SwinUNETR(
        in_channels=gcfg["in_channels"],
        out_channels=gcfg["out_channels"],
        feature_size=gcfg["feature_size"],
        spatial_dims=gcfg.get("spatial_dims", 3),
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    generator.load_state_dict(state_dict)
    generator.eval()
    print(f"Loaded generator from {checkpoint_path}")

    # Load data list
    data_list = build_data_list(
        config["data_root"], config["train_dir"], config["modalities"], has_labels=True
    )
    if max_cases is not None:
        data_list = data_list[:max_cases]
    print(f"Generating synthetic data from {len(data_list)} cases, {num_synthetic} per case")

    # Collect all tumour labels for random selection
    all_labels = []
    for entry in data_list:
        lbl = nib.load(entry["label"]).get_fdata().astype(np.float32)
        all_labels.append(lbl)

    generated = 0
    with torch.no_grad():
        for i, entry in enumerate(data_list):
            case_id = entry["case_id"]

            # Load full-resolution scan
            scan_vols = []
            for path in entry["image"]:
                vol = nib.load(path).get_fdata().astype(np.float32)
                scan_vols.append(vol)
            full_scan = np.stack(scan_vols, axis=0)  # (4, D, H, W)
            full_label = all_labels[i]

            ref_nii = nib.load(entry["image"][0])

            for j in range(num_synthetic):
                # Pick a random tumour label from a DIFFERENT case
                donor_idx = random.choice([k for k in range(len(data_list)) if k != i])
                donor_label = all_labels[donor_idx]

                # Extract tumour from donor
                tumour_crop, _ = extract_tumour_label(donor_label, crop_size)
                if tumour_crop is None:
                    continue

                # Find a healthy location in the target scan
                location = find_healthy_location(full_label, crop_size)
                if location is None:
                    continue

                ci, cj, ck = location
                hd, hh, hw = crop_size[0] // 2, crop_size[1] // 2, crop_size[2] // 2

                # Extract the healthy patch from target scan
                patch_scan = full_scan[:, ci - hd:ci + hd, cj - hh:cj + hh, ck - hw:ck + hw].copy()

                # Normalize patch to [-1, 1]
                for c in range(4):
                    vmin, vmax = patch_scan[c].min(), patch_scan[c].max()
                    if vmax - vmin > 0:
                        patch_scan[c] = 2.0 * (patch_scan[c] - vmin) / (vmax - vmin) - 1.0

                # Create noisy input with the donor tumour label
                noisy_scan = create_noisy_input(patch_scan, tumour_crop, crop_size)
                label_norm = tumour_crop[np.newaxis] / max(tumour_crop.max(), 1.0)
                noisy_input = np.concatenate([noisy_scan, label_norm], axis=0)  # (5, 96, 96, 96)

                # Run generator
                inp = torch.from_numpy(noisy_input).unsqueeze(0).to(device)
                fake_patch = generator(inp).squeeze(0).cpu().numpy()  # (4, 96, 96, 96)

                # Denormalize fake patch back to original intensity range
                for c in range(4):
                    orig_patch = full_scan[c, ci - hd:ci + hd, cj - hh:cj + hh, ck - hw:ck + hw]
                    omin, omax = orig_patch.min(), orig_patch.max()
                    fake_patch[c] = (fake_patch[c] + 1.0) / 2.0 * (omax - omin) + omin

                # Create synthetic scan: paste generated region into copy of original
                synth_scan = full_scan.copy()
                synth_scan[:, ci - hd:ci + hd, cj - hh:cj + hh, ck - hw:ck + hw] = fake_patch

                # Create synthetic label: place donor tumour at location
                synth_label = full_label.copy()
                synth_label[ci - hd:ci + hd, cj - hh:cj + hh, ck - hw:ck + hw] = tumour_crop

                # Save synthetic case
                synth_id = f"{case_id}-synth-{j:03d}"
                case_dir = os.path.join(output_dir, synth_id)
                os.makedirs(case_dir, exist_ok=True)

                modalities = config["modalities"]
                for c, mod in enumerate(modalities):
                    nii = nib.Nifti1Image(synth_scan[c], affine=ref_nii.affine, header=ref_nii.header)
                    nib.save(nii, os.path.join(case_dir, f"{synth_id}-{mod}.nii.gz"))

                seg_nii = nib.Nifti1Image(synth_label.astype(np.uint8), affine=ref_nii.affine, header=ref_nii.header)
                nib.save(seg_nii, os.path.join(case_dir, f"{synth_id}-seg.nii.gz"))

                generated += 1

            if (i + 1) % 5 == 0 or (i + 1) == len(data_list):
                print(f"  [{i + 1}/{len(data_list)}] Generated {generated} synthetic cases so far")

    print(f"Generation complete. {generated} synthetic cases saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic BraTS data using GliGAN")
    parser.add_argument("--config", type=str, default="config_gan.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint")
    parser.add_argument("--output_dir", type=str, default="./synthetic_data")
    parser.add_argument("--max_cases", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    generate_synthetic_data(config, args.checkpoint, args.output_dir, args.max_cases)


if __name__ == "__main__":
    main()
