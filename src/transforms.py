import numpy as np
import torch
from monai.transforms import MapTransform


class ConvertBraTSLabelsToRegions:
    """Convert BraTS integer labels to binary masks for the 6 evaluation regions.

    Input:  (H, W, D) array with values {0, 1, 2, 3, 4}
    Output: dict with keys ET, NETC, SNFH, RC, TC, WT, each a binary (H, W, D) array.
    """

    REGION_LABELS = {
        "ET": {3},
        "NETC": {1},
        "SNFH": {2},
        "RC": {4},
        "TC": {1, 3},
        "WT": {1, 2, 3},
    }

    def __call__(self, seg):
        regions = {}
        for name, labels in self.REGION_LABELS.items():
            mask = np.zeros_like(seg, dtype=np.uint8)
            for lbl in labels:
                mask[seg == lbl] = 1
            regions[name] = mask
        return regions


class AnatomicalConstraints:
    """
    Enforce ET subset-of-TC anatomical constraint at post-processing time.

    Any ET (label 3) connected component that has no NETC (label 1) neighbour
    within dilation_radius voxels is reclassified as NETC. Isolated ET blobs
    are almost always false-positive predictions: the tumour core is present
    but the model over-predicts the enhancing sub-region.
    """

    def __init__(self, dilation_radius: int = 3):
        self.dilation_radius = dilation_radius

    def __call__(self, seg: np.ndarray) -> np.ndarray:
        import cc3d
        from scipy.ndimage import binary_dilation

        seg = seg.copy()
        et_mask = seg == 3
        if et_mask.sum() == 0:
            return seg

        netc_mask = seg == 1
        r = self.dilation_radius
        struct = np.ones((2 * r + 1, 2 * r + 1, 2 * r + 1), dtype=bool)
        netc_dilated = binary_dilation(netc_mask, structure=struct)

        et_comps = cc3d.connected_components(et_mask.astype(np.uint8), connectivity=26)
        for comp_id in range(1, int(et_comps.max()) + 1):
            comp_mask = et_comps == comp_id
            if not np.any(comp_mask & netc_dilated):
                seg[comp_mask] = 1  # reclassify isolated ET as NETC
        return seg


class RemoveSmallConnectedComponents(MapTransform):
    """Remove connected components smaller than min_size voxels from each
    foreground class in the predicted segmentation."""

    def __init__(self, keys, min_size=20):
        super().__init__(keys)
        self.min_size = min_size

    def __call__(self, data):
        import cc3d

        d = dict(data)
        for key in self.keys:
            seg = d[key]
            if isinstance(seg, torch.Tensor):
                seg = seg.numpy()
            seg = seg.copy()
            for label_val in [1, 2, 3, 4]:
                binary = (seg == label_val).astype(np.uint8)
                if binary.sum() == 0:
                    continue
                components = cc3d.connected_components(binary, connectivity=26)
                for comp_id in range(1, components.max() + 1):
                    comp_mask = components == comp_id
                    if comp_mask.sum() < self.min_size:
                        seg[comp_mask] = 0
            d[key] = seg
        return d
