"""
Ensemble inference across all three model types:
  - HierarchicalSegNet  (2D slice-by-slice)
  - DynUNet             (3D sliding window)
  - SwinUNETR           (3D sliding window)

Each model produces a (5, H, W, D) soft probability map.
These are averaged and argmax-ed to produce the final segmentation.

Usage:
    python ensemble_all_models.py \\
        --configs config_hierarchical_2d.yaml config_dynunet.yaml config_swinunetr.yaml \\
        --checkpoints checkpoints_hierarchical/best_model_hierarchical.pth \\
                      checkpoints_dynunet/best_model.pth \\
                      checkpoints_swinunetr/best_model.pth \\
        --input_dir validation_data \\
        --output_dir ./predictions_ensemble_all
"""
import argparse
import json
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
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
from src.transforms import RemoveSmallConnectedComponents
from src.utils import get_device, load_checkpoint, load_config


_LOAD_TRANSFORM = Compose([
    LoadImaged(keys=["image"], image_only=True),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image"]),
])


def _get_3d_probs(model, volume: np.ndarray, device, roi_size, sw_batch_size, overlap, sw_mode):
    """
    3-D sliding window inference.
    volume : (4, H, W, D) float32
    Returns (5, H, W, D) float32 probability map.
    """
    x = torch.from_numpy(volume).unsqueeze(0).float().to(device)  # (1, 4, H, W, D)
    with torch.no_grad():
        out = sliding_window_inference(
            x,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode=sw_mode,
        )
    return torch.softmax(out, dim=1)[0].cpu().numpy()  # (5, H, W, D)


def _get_hierarchical_probs(model, volume: np.ndarray, device):
    """
    2-D slice-by-slice inference for HierarchicalSegNet.
    Converts the 4 binary head outputs into a normalised 5-class
    probability map so it can be averaged with the 3-D models.

    volume : (4, H, W, D) float32
    Returns (5, H, W, D) float32 probability map.
    """
    H, W, D = volume.shape[1], volume.shape[2], volume.shape[3]
    pad_h = (4 - H % 4) % 4
    pad_w = (4 - W % 4) % 4
    probs = np.zeros((5, H, W, D), dtype=np.float32)

    with torch.no_grad():
        for z in range(D):
            x = torch.from_numpy(
                volume[:, :, :, z].copy()
            ).unsqueeze(0).float().to(device)  # (1, 4, H, W)

            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h))

            wt_l, tc_l, et_l, rc_l = model(x)

            p_wt = torch.softmax(wt_l, dim=1)[0, 1, :H, :W].cpu().numpy()  # (H, W)
            p_tc = torch.softmax(tc_l, dim=1)[0, 1, :H, :W].cpu().numpy()
            p_et = torch.softmax(et_l, dim=1)[0, 1, :H, :W].cpu().numpy()
            p_rc = torch.softmax(rc_l, dim=1)[0, 1, :H, :W].cpu().numpy()

            # Decompose into 5-class soft probabilities via the hierarchy
            p_snfh = p_wt * (1.0 - p_tc)
            p_netc = p_wt * p_tc * (1.0 - p_et)
            p_et_v = p_wt * p_tc * p_et
            p_bg   = (1.0 - p_wt) * (1.0 - p_rc)

            # channel order: BG=0, NETC=1, SNFH=2, ET=3, RC=4
            s = np.stack([p_bg, p_netc, p_snfh, p_et_v, p_rc], axis=0)
            s = s / (s.sum(axis=0, keepdims=True) + 1e-8)
            probs[:, :, :, z] = s

    return probs  # (5, H, W, D)


def _apply_weights(prob_maps: list, entries: list, weights: dict) -> np.ndarray:
    """
    Region-specific weighted average of probability maps.

    For each class c, each model's contribution is scaled by its
    pre-computed Dice weight for that class (weights sum to 1 per class).

    prob_maps : list of (5, H, W, D) arrays, one per model
    entries   : list of (model, cfg, mode) tuples — used to look up model names
    weights   : { model_name: [w_bg, w_netc, w_snfh, w_et, w_rc] }
    """
    combined = np.zeros_like(prob_maps[0])  # (5, H, W, D)
    for (_, cfg, _), probs in zip(entries, prob_maps):
        model_name = cfg["model"]["name"]
        w = np.array(weights[model_name], dtype=np.float32)  # (5,)
        combined += probs * w.reshape(5, 1, 1, 1)
    return combined  # already weighted-summed; weights per class sum to 1


def run_ensemble(configs, checkpoint_paths, input_dir, output_dir,
                 weights=None, min_size=20, max_cases=None):
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    ref_config = configs[0]
    has_labels = input_dir == ref_config.get("train_dir", "training_data1_v2")
    data_list = build_data_list(
        ref_config["data_root"], input_dir, ref_config["modalities"], has_labels=has_labels
    )
    if max_cases is not None:
        data_list = data_list[:max_cases]

    strategy = "region-specific weighted averaging" if weights else "simple averaging"
    print(f"Ensemble of {len(configs)} models on {len(data_list)} cases [{strategy}]")

    # Load all models and tag each by inference mode
    entries = []
    for cfg, ckpt in zip(configs, checkpoint_paths):
        model = build_model(cfg).to(device)
        load_checkpoint(ckpt, model)
        model.eval()
        model_name = cfg["model"]["name"]
        mode = "hierarchical" if model_name == "HierarchicalSegNet" else "3d"
        entries.append((model, cfg, mode))
        print(f"  Loaded {model_name} ({mode}) from {ckpt}")

    post_proc = RemoveSmallConnectedComponents(keys=["pred"], min_size=min_size)

    for i, case in enumerate(data_list):
        case_id = case["case_id"]

        # Load the 3-D volume once — shared across all models
        sample = _LOAD_TRANSFORM({"image": case["image"]})
        img_tensor = sample["image"]
        volume = np.array(img_tensor, dtype=np.float32)  # (4, H, W, D)

        prob_maps = []
        for model, cfg, mode in entries:
            if mode == "hierarchical":
                probs = _get_hierarchical_probs(model, volume, device)
            else:
                inf = cfg["inference"]
                probs = _get_3d_probs(
                    model, volume, device,
                    roi_size=tuple(inf["roi_size"]),
                    sw_batch_size=inf["sw_batch_size"],
                    overlap=inf["overlap"],
                    sw_mode=inf["mode"],
                )
            prob_maps.append(probs)

        # Combine: region-specific weighted average or simple mean
        if weights:
            avg = _apply_weights(prob_maps, entries, weights)
        else:
            avg = np.mean(prob_maps, axis=0)

        pred = avg.argmax(axis=0).astype(np.uint8)  # (H, W, D)

        # Remove tiny false-positive blobs
        result = post_proc({"pred": pred})
        pred = result["pred"]

        # Save NIfTI
        if hasattr(img_tensor, "affine"):
            affine = img_tensor.affine.numpy()
        else:
            affine = nib.load(case["image"][0]).affine

        nib.save(
            nib.Nifti1Image(pred, affine=affine),
            os.path.join(output_dir, f"{case_id}.nii.gz"),
        )

        if (i + 1) % 5 == 0 or (i + 1) == len(data_list):
            print(f"  [{i+1}/{len(data_list)}] Saved {case_id}")

    print(f"Done. Predictions saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="BRaTS ensemble: HierarchicalSegNet + AttentionUnet + SwinUNETR")
    parser.add_argument("--configs",     nargs="+", required=True, help="Config YAML for each model")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint for each model (same order)")
    parser.add_argument("--input_dir",   default="validation_data")
    parser.add_argument("--output_dir",  default="./predictions_ensemble_all")
    parser.add_argument("--weights",     default=None,
                        help="Path to ensemble_weights.json from compute_weights.py. "
                             "If omitted, falls back to simple averaging.")
    parser.add_argument("--min_size",    type=int, default=20)
    parser.add_argument("--max_cases",   type=int, default=None)
    args = parser.parse_args()

    if len(args.configs) != len(args.checkpoints):
        raise ValueError("Number of configs must match number of checkpoints")

    configs = [load_config(c) for c in args.configs]

    weights = None
    if args.weights:
        with open(args.weights) as f:
            weights = json.load(f)
        print(f"Loaded weights from {args.weights}")

    run_ensemble(configs, args.checkpoints, args.input_dir, args.output_dir,
                 weights=weights, min_size=args.min_size, max_cases=args.max_cases)


if __name__ == "__main__":
    main()
