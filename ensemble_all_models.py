"""
Ensemble inference across all three model types:
  - HierarchicalSegNet  (2D slice-by-slice)
  - AttentionUnet       (3D sliding window)
  - SwinUNETR           (3D sliding window)

Each model produces a (5, H, W, D) soft probability map.
Three ensemble strategies are supported:
  simple        — plain mean across models (default)
  dice_weighted — region-specific weighted average from a JSON file of
                  per-class Dice scores computed on the validation set
  uncertainty   — per-voxel entropy-weighted average (low-confidence
                  model predictions contribute less to the final answer)

Usage:
    python ensemble_all_models.py \\
        --configs config_hierarchical_2d.yaml config_attention_unet.yaml config_swinunetr.yaml \\
        --checkpoints checkpoints_hierarchical/best_model_hierarchical.pth \\
                      checkpoints_attention_unet/best_model.pth \\
                      checkpoints_swinunetr/best_model.pth \\
        --input_dir validation_data \\
        --output_dir ./predictions_ensemble_all \\
        --weighting uncertainty \\
        --tta
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
from src.transforms import AnatomicalConstraints, RemoveSmallConnectedComponents
from src.tta import tta_hierarchical_slice, tta_sliding_window
from src.utils import get_device, load_checkpoint, load_config


_LOAD_TRANSFORM = Compose([
    LoadImaged(keys=["image"], image_only=True),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image"]),
])


def _get_3d_probs(model, volume: np.ndarray, device, roi_size, sw_batch_size, overlap, sw_mode,
                  use_tta: bool = False):
    """
    3-D sliding window inference (optionally with 8-flip TTA).
    volume : (4, H, W, D) float32
    Returns (5, H, W, D) float32 probability map.
    """
    x = torch.from_numpy(volume).unsqueeze(0).float().to(device)  # (1, 4, H, W, D)
    with torch.no_grad():
        if use_tta:
            probs = tta_sliding_window(model, x, roi_size, sw_batch_size, overlap, sw_mode)
            return probs[0].cpu().numpy()  # (5, H, W, D)
        else:
            out = sliding_window_inference(
                x, roi_size=roi_size, sw_batch_size=sw_batch_size,
                predictor=model, overlap=overlap, mode=sw_mode,
            )
            return torch.softmax(out, dim=1)[0].cpu().numpy()  # (5, H, W, D)


def _get_hierarchical_probs(model, volume: np.ndarray, device, use_tta: bool = False):
    """
    2-D slice-by-slice inference for HierarchicalSegNet (optionally with 4-flip TTA).
    Converts the 4 binary head outputs into a normalised 5-class probability map.

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

            if use_tta:
                p_wt, p_tc, p_et, p_rc = tta_hierarchical_slice(model, x, H, W)
            else:
                wt_l, tc_l, et_l, rc_l = model(x)
                p_wt = torch.softmax(wt_l, dim=1)[0, 1, :H, :W].cpu().numpy()
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


def _apply_dice_weights(prob_maps: list, entries: list, weights: dict) -> np.ndarray:
    """
    Region-specific weighted average using pre-computed per-class Dice scores.

    weights : { model_name: [w_bg, w_netc, w_snfh, w_et, w_rc] }
              (weights per class should sum to 1 across models)
    """
    combined = np.zeros_like(prob_maps[0])  # (5, H, W, D)
    for (_, cfg, _), p in zip(entries, prob_maps):
        w = np.array(weights[cfg["model"]["name"]], dtype=np.float32)  # (5,)
        combined += p * w.reshape(5, 1, 1, 1)
    return combined


def _uncertainty_ensemble(prob_maps: list) -> np.ndarray:
    """
    Per-voxel Shannon entropy-weighted ensemble.

    Models that are uncertain (high entropy softmax) contribute less.
    Weight for model m at voxel v: 1 / (entropy_m(v) + eps).
    Weights are normalised across models per voxel.

    prob_maps : list of (5, H, W, D) float32 arrays
    Returns   : (5, H, W, D) weighted probability map
    """
    eps = 1e-8
    weights = []
    for p in prob_maps:
        entropy = -(p * np.log(p + eps)).sum(axis=0, keepdims=True)  # (1, H, W, D)
        weights.append(1.0 / (entropy + eps))

    total_w = sum(weights)  # (1, H, W, D)
    combined = np.zeros_like(prob_maps[0])
    for p, w in zip(prob_maps, weights):
        combined += p * (w / (total_w + eps))
    return combined


def run_ensemble(
    configs, checkpoint_paths, input_dir, output_dir,
    weights=None, weighting="simple",
    min_size=20, max_cases=None,
    use_tta=False, anatomy_fix=True,
):
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    ref_config = configs[0]
    has_labels = input_dir == ref_config.get("train_dir", "training_data1_v2")
    data_list = build_data_list(
        ref_config["data_root"], input_dir, ref_config["modalities"], has_labels=has_labels
    )
    if max_cases is not None:
        data_list = data_list[:max_cases]

    tta_str = " +TTA" if use_tta else ""
    print(f"Ensemble of {len(configs)} models | {weighting} weighting{tta_str} | {len(data_list)} cases")

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
    anatomy = AnatomicalConstraints(dilation_radius=3) if anatomy_fix else None

    for i, case in enumerate(data_list):
        case_id = case["case_id"]

        sample = _LOAD_TRANSFORM({"image": case["image"]})
        img_tensor = sample["image"]
        volume = np.array(img_tensor, dtype=np.float32)  # (4, H, W, D)

        prob_maps = []
        for model, cfg, mode in entries:
            if mode == "hierarchical":
                p = _get_hierarchical_probs(model, volume, device, use_tta=use_tta)
            else:
                inf = cfg["inference"]
                p = _get_3d_probs(
                    model, volume, device,
                    roi_size=tuple(inf["roi_size"]),
                    sw_batch_size=inf["sw_batch_size"],
                    overlap=inf["overlap"],
                    sw_mode=inf["mode"],
                    use_tta=use_tta,
                )
            prob_maps.append(p)

        if weighting == "uncertainty":
            avg = _uncertainty_ensemble(prob_maps)
        elif weighting == "dice_weighted" and weights is not None:
            avg = _apply_dice_weights(prob_maps, entries, weights)
        else:
            avg = np.mean(prob_maps, axis=0)

        pred = avg.argmax(axis=0).astype(np.uint8)  # (H, W, D)

        result = post_proc({"pred": pred})
        pred = result["pred"]

        if anatomy is not None:
            pred = anatomy(pred)

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
    parser = argparse.ArgumentParser(
        description="BRaTS ensemble: HierarchicalSegNet + AttentionUnet + SwinUNETR"
    )
    parser.add_argument("--configs",     nargs="+", required=True, help="Config YAML for each model")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint for each model (same order)")
    parser.add_argument("--input_dir",   default="validation_data")
    parser.add_argument("--output_dir",  default="./predictions_ensemble_all")
    parser.add_argument("--weighting",   default="simple",
                        choices=["simple", "dice_weighted", "uncertainty"],
                        help="Ensemble strategy: simple mean, Dice-score-weighted, or uncertainty-weighted")
    parser.add_argument("--weights",     default=None,
                        help="Path to ensemble_weights.json (required for --weighting dice_weighted)")
    parser.add_argument("--min_size",    type=int, default=20)
    parser.add_argument("--max_cases",   type=int, default=None)
    parser.add_argument("--tta",         action="store_true",
                        help="Enable test-time augmentation (8-flip 3D / 4-flip 2D)")
    parser.add_argument("--no_anatomy_fix", action="store_true",
                        help="Disable anatomical constraint post-processing")
    args = parser.parse_args()

    if len(args.configs) != len(args.checkpoints):
        raise ValueError("Number of configs must match number of checkpoints")

    configs = [load_config(c) for c in args.configs]

    weights = None
    if args.weights:
        with open(args.weights) as f:
            weights = json.load(f)
        print(f"Loaded dice weights from {args.weights}")

    if args.weighting == "dice_weighted" and weights is None:
        raise ValueError("--weighting dice_weighted requires --weights <path-to-json>")

    run_ensemble(
        configs, args.checkpoints, args.input_dir, args.output_dir,
        weights=weights, weighting=args.weighting,
        min_size=args.min_size, max_cases=args.max_cases,
        use_tta=args.tta, anatomy_fix=not args.no_anatomy_fix,
    )


if __name__ == "__main__":
    main()
