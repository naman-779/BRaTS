import torch
import torch.nn as nn
from monai.losses import DiceCELoss


def build_loss(config):
    lcfg = config["loss"]
    return DiceCELoss(
        softmax=lcfg["softmax"],
        to_onehot_y=lcfg["to_onehot_y"],
        include_background=lcfg["include_background"],
        lambda_dice=lcfg["lambda_dice"],
        lambda_ce=lcfg["lambda_ce"],
    )


class HierarchicalLoss(nn.Module):
    """
    Trains each of the 4 hierarchical heads independently with DiceCE loss.

    GT label map (values 0-4) is remapped to binary targets per head:
      wt  : 1 where label ∈ {1,2,3},  0 elsewhere
      tc  : 1 where label ∈ {1,3},    0 elsewhere
      et  : 1 where label == 3,        0 elsewhere
      rc  : 1 where label == 4,        0 elsewhere

    RC is upweighted by default because it is the rarest and hardest region.
    """

    def __init__(self, lambda_tc: float = 1.0, lambda_et: float = 1.0, lambda_rc: float = 2.0):
        super().__init__()
        self._dice_ce = DiceCELoss(softmax=True, to_onehot_y=True, include_background=False)
        self.lambda_tc = lambda_tc
        self.lambda_et = lambda_et
        self.lambda_rc = lambda_rc

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
        l_wt = self._dice_ce(wt_logits, wt_t)
        l_tc = self._dice_ce(tc_logits, tc_t)
        l_et = self._dice_ce(et_logits, et_t)
        l_rc = self._dice_ce(rc_logits, rc_t)
        return l_wt + self.lambda_tc * l_tc + self.lambda_et * l_et + self.lambda_rc * l_rc
