import numpy as np
import cc3d
from scipy.ndimage import binary_dilation, generate_binary_structure
from surface_distance import compute_surface_distances, compute_robust_hausdorff

# BRaTS 2024 evaluation regions
REGION_LABELS = {
    "ET": {3},
    "NETC": {1},
    "SNFH": {2},
    "RC": {4},
    "TC": {1, 3},
    "WT": {1, 2, 3},
}


def _labels_to_binary(seg, label_set):
    mask = np.zeros_like(seg, dtype=bool)
    for lbl in label_set:
        mask |= seg == lbl
    return mask


def dice_score(pred, gt):
    intersection = np.sum(pred & gt)
    total = np.sum(pred) + np.sum(gt)
    if total == 0:
        return 1.0  # Both empty = perfect agreement
    return 2.0 * intersection / total


def hausdorff_95(pred, gt, spacing_mm=(1.0, 1.0, 1.0)):
    if np.sum(pred) == 0 and np.sum(gt) == 0:
        return 0.0
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return 374.0  # Max possible distance for BRaTS volumes
    surface_distances = compute_surface_distances(
        gt.astype(bool), pred.astype(bool), spacing_mm
    )
    return compute_robust_hausdorff(surface_distances, 95)


def _get_lesion_components(binary_mask, dilation_iters=3):
    """Get connected components after optional dilation to group nearby lesions."""
    if binary_mask.sum() == 0:
        return binary_mask, np.zeros_like(binary_mask, dtype=np.int32)
    struct = generate_binary_structure(3, 2)  # 18-connectivity for dilation
    dilated = binary_dilation(binary_mask, structure=struct, iterations=dilation_iters)
    components = cc3d.connected_components(dilated.astype(np.uint8), connectivity=26)
    return dilated, components


def compute_lesionwise_dice(pred_seg, gt_seg, region_name, spacing_mm=(1.0, 1.0, 1.0)):
    """Compute lesion-wise Dice and HD95 for a single region.

    Following BraTS 2024 evaluation protocol:
    1. Convert to binary masks for the region.
    2. Dilate ground truth to group nearby lesion components.
    3. Find connected components in dilated ground truth.
    4. Match predicted lesions to ground truth lesions.
    5. Compute per-lesion Dice and HD95, penalizing unmatched components.

    Returns dict with 'dice' and 'hd95' (averaged across lesions).
    """
    label_set = REGION_LABELS[region_name]
    pred_binary = _labels_to_binary(pred_seg, label_set)
    gt_binary = _labels_to_binary(gt_seg, label_set)

    # Handle empty cases
    if gt_binary.sum() == 0 and pred_binary.sum() == 0:
        return {"dice": 1.0, "hd95": 0.0}
    if gt_binary.sum() == 0 and pred_binary.sum() > 0:
        return {"dice": 0.0, "hd95": 374.0}
    if gt_binary.sum() > 0 and pred_binary.sum() == 0:
        return {"dice": 0.0, "hd95": 374.0}

    # Get ground truth lesion components
    dilation_iters = 5 if region_name in ("ET", "NETC", "SNFH", "RC") else 3
    _, gt_components = _get_lesion_components(gt_binary, dilation_iters)

    gt_lesion_ids = set(np.unique(gt_components)) - {0}
    lesion_dices = []
    lesion_hd95s = []

    pred_matched = np.zeros_like(pred_binary, dtype=bool)

    for lesion_id in gt_lesion_ids:
        gt_lesion_mask = gt_components == lesion_id
        # Restrict to original (undilated) ground truth within this component region
        gt_lesion_actual = gt_binary & gt_lesion_mask
        pred_lesion_actual = pred_binary & gt_lesion_mask

        if gt_lesion_actual.sum() == 0:
            continue

        pred_matched |= pred_lesion_actual

        d = dice_score(pred_lesion_actual, gt_lesion_actual)
        h = hausdorff_95(pred_lesion_actual, gt_lesion_actual, spacing_mm)
        lesion_dices.append(d)
        lesion_hd95s.append(h)

    # Penalize false positive predictions not matched to any GT lesion
    fp_mask = pred_binary & ~pred_matched
    if fp_mask.sum() > 0:
        fp_components = cc3d.connected_components(fp_mask.astype(np.uint8), connectivity=26)
        num_fp = len(set(np.unique(fp_components)) - {0})
        for _ in range(num_fp):
            lesion_dices.append(0.0)
            lesion_hd95s.append(374.0)

    if not lesion_dices:
        return {"dice": 0.0, "hd95": 374.0}

    return {
        "dice": float(np.mean(lesion_dices)),
        "hd95": float(np.mean(lesion_hd95s)),
    }


def compute_all_metrics(pred_seg, gt_seg, spacing_mm=(1.0, 1.0, 1.0)):
    """Compute lesion-wise Dice and HD95 for all 6 BraTS 2024 regions."""
    results = {}
    for region_name in REGION_LABELS:
        results[region_name] = compute_lesionwise_dice(
            pred_seg, gt_seg, region_name, spacing_mm
        )
    return results
