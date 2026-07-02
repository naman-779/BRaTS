import torch
import torch.nn as nn
from monai.losses import DiceCELoss, HausdorffDTLoss


def build_loss(config):
    lcfg = config["loss"]
    base = DiceCELoss(
        softmax=lcfg["softmax"],
        to_onehot_y=lcfg["to_onehot_y"],
        include_background=lcfg["include_background"],
        lambda_dice=lcfg["lambda_dice"],
        lambda_ce=lcfg["lambda_ce"],
    )
    lambda_hd = lcfg.get("lambda_hausdorff", 0.0)
    if lambda_hd > 0.0:
        hd = HausdorffDTLoss(
            softmax=lcfg["softmax"],
            to_onehot_y=lcfg["to_onehot_y"],
            include_background=lcfg["include_background"],
        )
        return CombinedLoss(base, hd, lambda_hd)
    return base


class CombinedLoss(nn.Module):
    """DiceCE + lambda_hausdorff * HausdorffDT boundary-aware loss."""

    def __init__(self, dice_ce_loss, hd_loss, lambda_hd: float = 0.1):
        super().__init__()
        self.dice_ce = dice_ce_loss
        self.hd = hd_loss
        self.lambda_hd = lambda_hd

    def forward(self, pred, target):
        return self.dice_ce(pred, target) + self.lambda_hd * self.hd(pred, target)


class HierarchicalLoss(nn.Module):
    """
    Trains each of the 4 hierarchical heads independently with DiceCE loss,
    optionally augmented with a boundary-aware HausdorffDT term.

    GT label map (values 0-4) is remapped to binary targets per head:
      wt  : 1 where label in {1,2,3},  0 elsewhere
      tc  : 1 where label in {1,3},    0 elsewhere
      et  : 1 where label == 3,        0 elsewhere
      rc  : 1 where label == 4,        0 elsewhere

    RC is upweighted by default because it is the rarest and hardest region.
    lambda_hausdorff=0 disables the HD term with zero overhead.
    """

    def __init__(
        self,
        lambda_tc: float = 1.0,
        lambda_et: float = 1.0,
        lambda_rc: float = 2.0,
        lambda_hausdorff: float = 0.0,
    ):
        super().__init__()
        self._dice_ce = DiceCELoss(softmax=True, to_onehot_y=True, include_background=False)
        self.lambda_tc = lambda_tc
        self.lambda_et = lambda_et
        self.lambda_rc = lambda_rc
        self.lambda_hd = lambda_hausdorff
        if lambda_hausdorff > 0.0:
            self._hd = HausdorffDTLoss(softmax=True, to_onehot_y=True, include_background=False)
        else:
            self._hd = None

    def _head_loss(self, logits, target):
        loss = self._dice_ce(logits, target)
        if self._hd is not None:
            loss = loss + self.lambda_hd * self._hd(logits, target)
        return loss

    @staticmethod
    def _derive_targets(label: torch.Tensor):
        """label: (B, 1, H, W) int — returns four (B, 1, H, W) long binary targets."""
        wt = ((label == 1) | (label == 2) | (label == 3)).long()
        tc = ((label == 1) | (label == 3)).long()
        et = (label == 3).long()
        rc = (label == 4).long()
        return wt, tc, et, rc

    def forward(self, wt_logits, tc_logits, et_logits, rc_logits, label):
        wt_t, tc_t, et_t, rc_t = self._derive_targets(label)
        l_wt = self._head_loss(wt_logits, wt_t)
        l_tc = self._head_loss(tc_logits, tc_t)
        l_et = self._head_loss(et_logits, et_t)
        l_rc = self._head_loss(rc_logits, rc_t)
        return l_wt + self.lambda_tc * l_tc + self.lambda_et * l_et + self.lambda_rc * l_rc