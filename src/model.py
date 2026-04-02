from monai.networks.nets import SegResNet, SwinUNETR


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
    else:
        raise ValueError(f"Unknown model name: {name}")

    return model
