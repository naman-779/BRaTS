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
