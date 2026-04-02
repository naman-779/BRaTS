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


def run_inference(config, checkpoint_path, input_dir, output_dir, remove_small=True, min_size=20, max_cases=None):
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    # Determine if labels exist (training data vs validation data)
    has_labels = input_dir == config["train_dir"]
    data_list = build_data_list(
        config["data_root"], input_dir, config["modalities"], has_labels=has_labels
    )
    if max_cases is not None:
        data_list = data_list[:max_cases]
    print(f"Running inference on {len(data_list)} cases from {input_dir}")

    ds = Dataset(data=data_list, transform=get_inference_transforms())
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    # Load model
    model = build_model(config).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    roi_size = tuple(config["inference"]["roi_size"])
    sw_batch_size = config["inference"]["sw_batch_size"]
    overlap = config["inference"]["overlap"]
    sw_mode = config["inference"]["mode"]
    amp_enabled = config["training"]["amp"]

    # We need an original reference NIfTI to copy the affine/header
    # Load one to get the template
    ref_case = data_list[0]
    ref_nii = nib.load(ref_case["image"][0])
    affine = ref_nii.affine

    post_proc = RemoveSmallConnectedComponents(keys=["pred"], min_size=min_size) if remove_small else None

    with torch.no_grad():
        for i, batch in enumerate(loader):
            case_id = data_list[i]["case_id"]
            images = batch["image"].to(device)

            with torch.amp.autocast("cuda", enabled=amp_enabled and device.type == "cuda"):
                outputs = sliding_window_inference(
                    images,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode=sw_mode,
                )

            # Argmax to get integer labels
            pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Post-processing: remove small components
            if post_proc is not None:
                result = post_proc({"pred": pred})
                pred = result["pred"]

            # Save as NIfTI using the reference affine
            # Load the actual case's affine for precise alignment
            case_nii = nib.load(data_list[i]["image"][0])
            nii_img = nib.Nifti1Image(pred, affine=case_nii.affine, header=case_nii.header)
            out_path = os.path.join(output_dir, f"{case_id}.nii.gz")
            nib.save(nii_img, out_path)

            if (i + 1) % 10 == 0 or (i + 1) == len(data_list):
                print(f"  [{i + 1}/{len(data_list)}] Saved {case_id}")

    print(f"Inference complete. Predictions saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="BRaTS 2024 Baseline Inference")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--input_dir", type=str, default="validation_data",
        help="Subdirectory name under data_root (e.g. validation_data or training_data1_v2)",
    )
    parser.add_argument("--output_dir", type=str, default="./predictions")
    parser.add_argument("--no_postprocess", action="store_true", help="Skip small component removal")
    parser.add_argument("--min_size", type=int, default=20, help="Min component size in voxels")
    parser.add_argument("--max_cases", type=int, default=None, help="Limit number of cases for quick evaluation")
    args = parser.parse_args()

    config = load_config(args.config)
    run_inference(
        config,
        args.checkpoint,
        args.input_dir,
        args.output_dir,
        remove_small=not args.no_postprocess,
        min_size=args.min_size,
        max_cases=args.max_cases,
    )


if __name__ == "__main__":
    main()
