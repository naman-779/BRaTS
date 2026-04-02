import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd

from src.metrics import REGION_LABELS, compute_all_metrics


def main():
    parser = argparse.ArgumentParser(description="BRaTS 2024 Evaluation")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory with predicted NIfTI segmentations")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory with ground truth cases (e.g. training_data1_v2)")
    parser.add_argument("--output", type=str, default="results.csv", help="Output CSV path")
    args = parser.parse_args()

    pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.endswith(".nii.gz")])
    print(f"Found {len(pred_files)} prediction files")

    all_results = []
    for pred_file in pred_files:
        case_id = pred_file.replace(".nii.gz", "")
        gt_path = os.path.join(args.gt_dir, case_id, f"{case_id}-seg.nii.gz")

        if not os.path.exists(gt_path):
            print(f"  Skipping {case_id}: ground truth not found")
            continue

        pred_nii = nib.load(os.path.join(args.pred_dir, pred_file))
        gt_nii = nib.load(gt_path)

        pred_seg = pred_nii.get_fdata().astype(np.uint8)
        gt_seg = gt_nii.get_fdata().astype(np.uint8)

        spacing = tuple(gt_nii.header.get_zooms()[:3])
        metrics = compute_all_metrics(pred_seg, gt_seg, spacing_mm=spacing)

        row = {"case_id": case_id}
        for region in REGION_LABELS:
            row[f"{region}_dice"] = metrics[region]["dice"]
            row[f"{region}_hd95"] = metrics[region]["hd95"]
        all_results.append(row)

        if len(all_results) % 20 == 0:
            print(f"  Evaluated {len(all_results)}/{len(pred_files)} cases")

    df = pd.DataFrame(all_results)
    df.to_csv(args.output, index=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results ({len(all_results)} cases)")
    print(f"{'='*60}")
    for region in REGION_LABELS:
        mean_dice = df[f"{region}_dice"].mean()
        mean_hd95 = df[f"{region}_hd95"].mean()
        print(f"  {region:4s}  Dice={mean_dice:.4f}  HD95={mean_hd95:.2f}")
    print(f"{'='*60}")
    overall_dice = np.mean([df[f"{r}_dice"].mean() for r in REGION_LABELS])
    overall_hd95 = np.mean([df[f"{r}_hd95"].mean() for r in REGION_LABELS])
    print(f"  Mean  Dice={overall_dice:.4f}  HD95={overall_hd95:.2f}")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
