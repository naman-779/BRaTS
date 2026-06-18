import torch
import torch.nn as nn
from monai.networks.nets import SegResNet, SwinUNETR


class HierarchicalSegNet(nn.Module):
    """
    2D slice-by-slice hierarchical segmentation with 4 cascaded heads.

    Anatomy-motivated cascade:
      wt_head  : 4-ch MRI → WT binary  (whole tumour vs background)
      tc_head  : 4-ch MRI + 2-ch WT  → TC binary  (tumour core vs SNFH)
      et_head  : 4-ch MRI + 2-ch TC  → ET binary  (ET vs NETC)
      rc_head  : 4-ch MRI → RC binary  (resection cavity, independent)

    Each head is a lightweight 2D SegResNet.  During training the caller
    passes GT one-hot maps (teacher forcing) for wt_gt / tc_gt.  At
    inference both are left as None and the model's own softmax outputs
    are used instead.
    """

    def __init__(self, in_channels: int = 4, init_filters: int = 16, dropout_prob: float = 0.1):
        super().__init__()

        def _head(in_ch: int) -> nn.Module:
            return SegResNet(
                spatial_dims=2,
                in_channels=in_ch,
                out_channels=2,
                init_filters=init_filters,
                blocks_down=(1, 2, 2),
                blocks_up=(1, 1),
                dropout_prob=dropout_prob,
            )

        self.wt_head = _head(in_channels)
        self.tc_head = _head(in_channels + 2)
        self.et_head = _head(in_channels + 2)
        self.rc_head = _head(in_channels)

    def forward(self, x: torch.Tensor, wt_gt=None, tc_gt=None):
        """
        x      : (B, 4, H, W)
        wt_gt  : (B, 2, H, W) one-hot GT for teacher forcing, or None
        tc_gt  : (B, 2, H, W) one-hot GT for teacher forcing, or None

        Returns (wt_logits, tc_logits, et_logits, rc_logits), each (B, 2, H, W).
        """
        wt_logits = self.wt_head(x)

        wt_cond = wt_gt if wt_gt is not None else torch.softmax(wt_logits.detach(), dim=1)
        tc_logits = self.tc_head(torch.cat([x, wt_cond], dim=1))

        tc_cond = tc_gt if tc_gt is not None else torch.softmax(tc_logits.detach(), dim=1)
        et_logits = self.et_head(torch.cat([x, tc_cond], dim=1))

        rc_logits = self.rc_head(x)

        return wt_logits, tc_logits, et_logits, rc_logits


def combine_hierarchical_predictions(
    wt_logits: torch.Tensor,
    tc_logits: torch.Tensor,
    et_logits: torch.Tensor,
    rc_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Merge the 4 binary head outputs into a single (B, 1, H, W) label map.

    Label mapping: 0=BG, 1=NETC, 2=SNFH, 3=ET, 4=RC
    """
    wt = torch.softmax(wt_logits, dim=1)[:, 1] > 0.5
    tc = torch.softmax(tc_logits, dim=1)[:, 1] > 0.5
    et = torch.softmax(et_logits, dim=1)[:, 1] > 0.5
    rc = torch.softmax(rc_logits, dim=1)[:, 1] > 0.5

    pred = torch.zeros(wt.shape, dtype=torch.long, device=wt_logits.device)
    pred[wt & ~tc] = 2          # SNFH
    pred[wt & tc & ~et] = 1     # NETC
    pred[wt & tc & et] = 3      # ET
    pred[rc] = 4                # RC overrides WT

    return pred.unsqueeze(1)    # (B, 1, H, W)


def build_model(config):
    mcfg = config["model"]
    name = mcfg["name"]

    if name == "SegResNet":
        model = SegResNet(
            spatial_dims=mcfg["spatial_dims"],
            init_filters=mcfg["init_filters"],
            in_channels=mcfg["in_channels"],
            out_channels=mcfg["out_channels"],
            blocks_down=tuple(mcfg["blocks_down"]),
            blocks_up=tuple(mcfg["blocks_up"]),
            dropout_prob=mcfg["dropout_prob"],
            norm=(mcfg["norm"], {"num_groups": mcfg["num_groups"]}),
            upsample_mode=mcfg["upsample_mode"],
        )
    elif name == "SwinUNETR":
        model = SwinUNETR(
            in_channels=mcfg["in_channels"],
            out_channels=mcfg["out_channels"],
            feature_size=mcfg["feature_size"],
            spatial_dims=mcfg.get("spatial_dims", 3),
            drop_rate=mcfg.get("drop_rate", 0.0),
            attn_drop_rate=mcfg.get("attn_drop_rate", 0.0),
            use_checkpoint=mcfg.get("use_checkpoint", False),
        )
    elif name == "HierarchicalSegNet":
        model = HierarchicalSegNet(
            in_channels=mcfg.get("in_channels", 4),
            init_filters=mcfg.get("init_filters", 16),
            dropout_prob=mcfg.get("dropout_prob", 0.1),
        )
    elif name == "AttentionUnet":
        from monai.networks.nets import AttentionUnet
        model = AttentionUnet(
            spatial_dims=mcfg.get("spatial_dims", 3),
            in_channels=mcfg["in_channels"],
            out_channels=mcfg["out_channels"],
            channels=tuple(mcfg["channels"]),
            strides=tuple(mcfg["strides"]),
            dropout=mcfg.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown model name: {name}")

    return model
