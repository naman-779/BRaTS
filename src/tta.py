"""Test-Time Augmentation (TTA) helpers for BraTS inference."""
import torch

_FLIP_2D = [(f0, f1) for f0 in (False, True) for f1 in (False, True)]
_FLIP_3D = [(f0, f1, f2) for f0 in (False, True) for f1 in (False, True) for f2 in (False, True)]


def tta_sliding_window(model, x, roi_size, sw_batch_size, overlap, mode):
    """
    8-flip TTA for 3D models using sliding_window_inference.

    Flips the input along each combination of spatial axes, runs inference,
    flips the output back, and averages. This is the standard TTA procedure
    for volumetric medical image segmentation.

    x : (1, 4, H, W, D) tensor on device
    Returns (1, 5, H, W, D) averaged probability tensor (already softmax-ed).
    """
    from monai.inferers import sliding_window_inference

    preds = []
    for f0, f1, f2 in _FLIP_3D:
        xi = x.clone()
        if f0:
            xi = torch.flip(xi, [2])
        if f1:
            xi = torch.flip(xi, [3])
        if f2:
            xi = torch.flip(xi, [4])

        out = sliding_window_inference(
            xi, roi_size=roi_size, sw_batch_size=sw_batch_size,
            predictor=model, overlap=overlap, mode=mode,
        )
        prob = torch.softmax(out, dim=1)

        if f0:
            prob = torch.flip(prob, [2])
        if f1:
            prob = torch.flip(prob, [3])
        if f2:
            prob = torch.flip(prob, [4])
        preds.append(prob)

    return torch.stack(preds).mean(dim=0)  # (1, 5, H, W, D)


def tta_hierarchical_slice(model, x, H, W):
    """
    4-flip TTA for HierarchicalSegNet on a single 2D padded slice.

    Flips along H and W independently (4 combos), averages softmax outputs,
    then crops back to the original (H, W) before padding.

    model : HierarchicalSegNet already in eval mode
    x     : (1, 4, H_pad, W_pad) tensor on device — already padded
    H, W  : original (unpadded) spatial dims for output cropping

    Returns wt_p, tc_p, et_p, rc_p — each (H, W) numpy float32.
    """
    wt_acc, tc_acc, et_acc, rc_acc = [], [], [], []

    for f0, f1 in _FLIP_2D:
        xi = x.clone()
        if f0:
            xi = torch.flip(xi, [2])
        if f1:
            xi = torch.flip(xi, [3])

        wt_l, tc_l, et_l, rc_l = model(xi)

        wt_p = torch.softmax(wt_l, dim=1)[0, 1]
        tc_p = torch.softmax(tc_l, dim=1)[0, 1]
        et_p = torch.softmax(et_l, dim=1)[0, 1]
        rc_p = torch.softmax(rc_l, dim=1)[0, 1]

        if f0:
            wt_p = torch.flip(wt_p, [0])
            tc_p = torch.flip(tc_p, [0])
            et_p = torch.flip(et_p, [0])
            rc_p = torch.flip(rc_p, [0])
        if f1:
            wt_p = torch.flip(wt_p, [1])
            tc_p = torch.flip(tc_p, [1])
            et_p = torch.flip(et_p, [1])
            rc_p = torch.flip(rc_p, [1])

        wt_acc.append(wt_p[:H, :W])
        tc_acc.append(tc_p[:H, :W])
        et_acc.append(et_p[:H, :W])
        rc_acc.append(rc_p[:H, :W])

    return (
        torch.stack(wt_acc).mean(0).cpu().numpy(),
        torch.stack(tc_acc).mean(0).cpu().numpy(),
        torch.stack(et_acc).mean(0).cpu().numpy(),
        torch.stack(rc_acc).mean(0).cpu().numpy(),
    )
