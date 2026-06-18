"""
Compute per-class ensemble weights from individual model results CSVs.

Run this AFTER training all three models and evaluating each one separately
with evaluate.py. The weights are derived from each model's mean validation
Dice per sub-region and saved to a JSON file used by ensemble_all_models.py.

How it works
------------
For each segmentation class (NETC, SNFH, ET, RC), each model's mean Dice on
the validation set becomes its raw weight. Weights are then normalised per
class so they sum to 1 across models. A model that scores higher on SNFH will
contribute more to the final SNFH prediction than a model that scored lower.

Usage
-----
    # Evaluate each model individually first:
    python evaluate.py --pred_dir predictions_hierarchical --gt_dir "Brats GLI/training_data1_v2" --output results_hierarchical.csv
    python evaluate.py --pred_dir predictions_attention    --gt_dir "Brats GLI/training_data1_v2" --output results_attention.csv
    python evaluate.py --pred_dir predictions_swinunetr   --gt_dir "Brats GLI/training_data1_v2" --output results_swinunetr.csv

    # Then compute weights:
    python compute_weights.py \\
        --results  results_hierarchical.csv results_attention.csv results_swinunetr.csv \\
        --names    HierarchicalSegNet AttentionUnet SwinUNETR \\
        --output   ensemble_weights.json
"""
import argparse
import json

import numpy as np
import pandas as pd

# Maps evaluate.py column prefix → label index in the 5-class segmentation
_REGION_TO_CLASS = {
    "NETC": 1,
    "SNFH": 2,
    "ET":   3,
    "RC":   4,
}


def compute_weights(results_csvs: list, model_names: list, output_path: str) -> dict:
    """
    Derive normalised per-class ensemble weights from per-model results CSVs.

    Parameters
    ----------
    results_csvs  : paths to evaluate.py output CSVs, one per model
    model_names   : display names matching the model.name in each config
    output_path   : where to write the JSON weights file

    Returns
    -------
    dict  { model_name: [w_bg, w_netc, w_snfh, w_et, w_rc] }
    """
    n_models  = len(results_csvs)
    n_classes = 5

    # raw_weights shape: (n_models, n_classes)
    raw = np.zeros((n_models, n_classes), dtype=np.float64)

    for i, csv_path in enumerate(results_csvs):
        df = pd.read_csv(csv_path)
        for region, cls_idx in _REGION_TO_CLASS.items():
            col = f"{region}_dice"
            if col in df.columns:
                raw[i, cls_idx] = float(df[col].mean())
            else:
                print(f"  Warning: column '{col}' not found in {csv_path}, defaulting to 0")

    # Background (class 0) is never directly scored in BraTS; give equal weight
    raw[:, 0] = 1.0 / n_models

    # Normalise: per class, weights across models sum to 1
    col_sums = raw.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0          # avoid divide-by-zero for all-zero columns
    normalised = raw / col_sums            # shape (n_models, n_classes)

    weights = {
        name: normalised[i].tolist()
        for i, name in enumerate(model_names)
    }

    with open(output_path, "w") as f:
        json.dump(weights, f, indent=2)

    # Print a readable summary
    class_names = ["BG", "NETC", "SNFH", "ET", "RC"]
    print("\n── Raw mean Dice per model ──────────────────────")
    header = f"  {'Model':<25}" + "".join(f"{c:>8}" for c in class_names[1:])
    print(header)
    for i, name in enumerate(model_names):
        row = f"  {name:<25}" + "".join(f"{raw[i, c]:>8.4f}" for c in range(1, n_classes))
        print(row)

    print("\n── Normalised ensemble weights per class ────────")
    print(header)
    for name, w in weights.items():
        row = f"  {name:<25}" + "".join(f"{w[c]:>8.3f}" for c in range(1, n_classes))
        print(row)

    print(f"\nWeights saved to {output_path}")
    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Derive per-class ensemble weights from individual model evaluation CSVs"
    )
    parser.add_argument(
        "--results", nargs="+", required=True,
        help="evaluate.py output CSVs — one per model, same order as --names",
    )
    parser.add_argument(
        "--names", nargs="+", required=True,
        help="Model names matching model.name in each config (e.g. HierarchicalSegNet AttentionUnet SwinUNETR)",
    )
    parser.add_argument(
        "--output", default="ensemble_weights.json",
        help="Path to write the JSON weights file",
    )
    args = parser.parse_args()

    if len(args.results) != len(args.names):
        raise ValueError("--results and --names must have the same number of entries")

    compute_weights(args.results, args.names, args.output)


if __name__ == "__main__":
    main()
