import argparse
import os

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.data import Dataset, DataLoader

from src.data import build_data_list, get_inference_transforms
from src.model import build_model
from src.transforms import RemoveSmallConnectedComponents
from src.utils import get_device, load_checkpoint, load_config


def run_ensemble_inference(
    configs,
    checkpoint_paths,
    input_dir,
    output_dir,
    min_size=20,
    max_cases=None,
):
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    # Use the first config for data settings (all configs share the same data params)
    ref_config = configs[0]

    # Build data list
    has_labels = input_dir == ref_config["train_dir"]
    data_list = build_data_list(
        ref_config["data_root"], input_dir, ref_config["modalities"], has_labels=has_labels
    )
    if max_cases is not None:
        data_list = data_list[:max_cases]
    print(f"Running ensemble inference on {len(data_list)} cases from {input_dir}")
    print(f"Ensemble of {len(configs)} models")

    ds = Dataset(data=data_list, transform=get_inference_transforms())
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Load all models
    models = []
    for cfg, ckpt_path in zip(configs, checkpoint_paths):
        model = build_model(cfg).to(device)
        load_checkpoint(ckpt_path, model)
        model.eval()
        models.append(model)
        print(f"  Loaded {cfg['model']['name']} from {ckpt_path}")

    # Inference settings from first config
    roi_size = tuple(ref_config["inference"]["roi_size"])
    sw_batch_size = ref_config["inference"]["sw_batch_size"]
    overlap = ref_config["inference"]["overlap"]
    sw_mode = ref_config["inference"]["mode"]

    post_proc = RemoveSmallConnectedComponents(keys=["pred"], min_size=min_size)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            case_id = data_list[i]["case_id"]
            images = batch["image"].to(device)

            # Collect softmax probability maps from each model
            prob_maps = []
            for model in models:
                outputs = sliding_window_inference(
                    images,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode=sw_mode,
                )
                # Apply softmax to get probabilities
                probs = torch.softmax(outputs, dim=1)
                prob_maps.append(probs)

            # Average probability maps across all models
            avg_probs = torch.stack(prob_maps, dim=0).mean(dim=0)

            # Argmax to get final segmentation
            pred = torch.argmax(avg_probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Post-processing: remove small connected components
            result = post_proc({"pred": pred})
            pred = result["pred"]

            # Save as NIfTI
            case_nii = nib.load(data_list[i]["image"][0])
            nii_img = nib.Nifti1Image(pred, affine=case_nii.affine, header=case_nii.header)
            out_path = os.path.join(output_dir, f"{case_id}.nii.gz")
            nib.save(nii_img, out_path)

            if (i + 1) % 5 == 0 or (i + 1) == len(data_list):
                print(f"  [{i + 1}/{len(data_list)}] Saved {case_id}")

    print(f"Ensemble inference complete. Predictions saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="BRaTS Ensemble Inference (average of N models)")
    parser.add_argument(
        "--configs", type=str, nargs="+", required=True,
        help="Config YAML files for each model",
    )
    parser.add_argument(
        "--checkpoints", type=str, nargs="+", required=True,
        help="Checkpoint paths for each model (same order as configs)",
    )
    parser.add_argument(
        "--input_dir", type=str, default="training_data1_v2",
        help="Subdirectory name under data_root",
    )
    parser.add_argument("--output_dir", type=str, default="./predictions_ensemble")
    parser.add_argument("--min_size", type=int, default=20, help="Min component size in voxels")
    parser.add_argument("--max_cases", type=int, default=None, help="Limit number of cases")
    args = parser.parse_args()

    if len(args.configs) != len(args.checkpoints):
        raise ValueError("Number of configs must match number of checkpoints")

    configs = [load_config(c) for c in args.configs]

    run_ensemble_inference(
        configs,
        args.checkpoints,
        args.input_dir,
        args.output_dir,
        min_size=args.min_size,
        max_cases=args.max_cases,
    )


if __name__ == "__main__":
    main()
