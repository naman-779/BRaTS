"""
Slice-by-slice inference for the hierarchical 2-D segmentation model.

Loads each 3-D volume, runs the model on every axial slice, reassembles the
predictions into a 3-D NIfTI file, and saves it to the output directory.
The saved files are compatible with evaluate.py.

Usage:
    python inference_hierarchical.py \\
        --config config_hierarchical_2d.yaml \\
        --checkpoint checkpoints_hierarchical/best_model_hierarchical.pth \\
        --input_dir validation_data \\
        --output_dir ./predictions_hierarchical
"""
import argparse
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
)

from src.data import build_data_list
from src.model import build_model
from src.utils import get_device, load_checkpoint, load_config


_LOAD_TRANSFORM = None


def _get_load_transform():
    global _LOAD_TRANSFORM
    if _LOAD_TRANSFORM is None:
        _LOAD_TRANSFORM = Compose([
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ])
    return _LOAD_TRANSFORM


def _combine_slice(wt_prob, tc_prob, et_prob, rc_prob) -> np.ndarray:
    """
    Merge per-head softmax probabilities into a single (H, W) label map.

    wt_prob, tc_prob, et_prob, rc_prob : (H, W) numpy float32, probability of
    the *positive* class (channel 1) for each head.

    Label mapping: 0=BG, 1=NETC, 2=SNFH, 3=ET, 4=RC
    """
    wt = wt_prob > 0.5
    tc = tc_prob > 0.5
    et = et_prob > 0.5
    rc = rc_prob > 0.5

    pred = np.zeros_like(wt, dtype=np.uint8)
    pred[wt & ~tc] = 2          # SNFH
    pred[wt & tc & ~et] = 1     # NETC
    pred[wt & tc & et] = 3      # ET
    pred[rc] = 4                # RC overrides WT labels
    return pred


@torch.no_grad()
def infer_volume(model, volume: np.ndarray, device: torch.device) -> np.ndarray:
    """
    volume : (4, H, W, D) float32 numpy array in RAS space.
    Returns (H, W, D) uint8 segmentation in the same space.
    """
    H, W, D = volume.shape[1], volume.shape[2], volume.shape[3]
    pad_h = (4 - H % 4) % 4
    pad_w = (4 - W % 4) % 4
    pred_3d = np.zeros((H, W, D), dtype=np.uint8)

    for z in range(D):
        x = torch.from_numpy(volume[:, :, :, z].copy()).unsqueeze(0).float().to(device)

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        wt_logits, tc_logits, et_logits, rc_logits = model(x)

        wt_p = torch.softmax(wt_logits, dim=1)[0, 1, :H, :W].cpu().numpy()
        tc_p = torch.softmax(tc_logits, dim=1)[0, 1, :H, :W].cpu().numpy()
        et_p = torch.softmax(et_logits, dim=1)[0, 1, :H, :W].cpu().numpy()
        rc_p = torch.softmax(rc_logits, dim=1)[0, 1, :H, :W].cpu().numpy()

        pred_3d[:, :, z] = _combine_slice(wt_p, tc_p, et_p, rc_p)

    return pred_3d


def run_inference(config, checkpoint_path, input_dir, output_dir, max_cases=None):
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    has_labels = input_dir == config.get("train_dir", "training_data1_v2")
    data_list = build_data_list(
        config["data_root"], input_dir, config["modalities"], has_labels=has_labels
    )
    if max_cases is not None:
        data_list = data_list[:max_cases]

    model = build_model(config).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Running inference on {len(data_list)} cases from {input_dir}")

    load_tfm = _get_load_transform()

    for i, case in enumerate(data_list):
        case_id = case["case_id"]

        # Load and preprocess the 3-D volume
        sample = load_tfm({"image": case["image"]})
        img_tensor = sample["image"]
        volume = np.array(img_tensor, dtype=np.float32)     # (4, H, W, D)

        # Infer slice-by-slice
        pred_3d = infer_volume(model, volume, device)       # (H, W, D)

        # Save NIfTI — affine from the MetaTensor tracks the RAS reorientation
        if hasattr(img_tensor, "affine"):
            affine = img_tensor.affine.numpy()
        else:
            affine = nib.load(case["image"][0]).affine

        out_nii = nib.Nifti1Image(pred_3d, affine=affine)
        out_path = os.path.join(output_dir, f"{case_id}.nii.gz")
        nib.save(out_nii, out_path)

        if (i + 1) % 5 == 0 or (i + 1) == len(data_list):
            print(f"  [{i+1}/{len(data_list)}] Saved {case_id}")

    print(f"Inference complete. Predictions saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Hierarchical 2-D BRaTS Inference")
    parser.add_argument("--config",     default="config_hierarchical_2d.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--input_dir",  default="validation_data",
                        help="Subdirectory under data_root (e.g. validation_data)")
    parser.add_argument("--output_dir", default="./predictions_hierarchical")
    parser.add_argument("--max_cases",  type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_inference(config, args.checkpoint, args.input_dir, args.output_dir, args.max_cases)


if __name__ == "__main__":
    main()
