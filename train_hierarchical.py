"""
Training script for the 2-D slice-by-slice hierarchical segmentation model.

Usage:
    python train_hierarchical.py --config config_hierarchical_2d.yaml
    python train_hierarchical.py --config config_hierarchical_2d.yaml --resume checkpoints_hierarchical/best_model_hierarchical.pth
"""
import argparse
import os
import time

import torch
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter

from src.data import get_dataloaders_2d
from src.losses import HierarchicalLoss
from src.model import build_model, combine_hierarchical_predictions
from src.scheduler import WarmupCosineScheduler
from src.utils import get_device, load_checkpoint, load_config, save_checkpoint, set_seed, setup_logging

CLASS_NAMES = ["NETC", "SNFH", "ET", "RC"]


def _teacher_forcing_inputs(labels: torch.Tensor, device: torch.device):
    """
    Derive one-hot GT conditioning tensors from label map for teacher forcing.

    labels : (B, 1, H, W)  int, values 0-4
    Returns (wt_gt, tc_gt) each (B, 2, H, W) float one-hot.
    """
    wt_target = ((labels == 1) | (labels == 2) | (labels == 3)).long()
    tc_target = ((labels == 1) | (labels == 3)).long()
    wt_gt = one_hot(wt_target, num_classes=2).float().to(device)
    tc_gt = one_hot(tc_target, num_classes=2).float().to(device)
    return wt_gt, tc_gt


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    epoch_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)          # (B, 4, H, W)
        labels = batch["label"].long().to(device)   # (B, 1, H, W)

        wt_gt, tc_gt = _teacher_forcing_inputs(labels, device)

        optimizer.zero_grad(set_to_none=True)
        wt_logits, tc_logits, et_logits, rc_logits = model(images, wt_gt=wt_gt, tc_gt=tc_gt)
        loss = loss_fn(wt_logits, tc_logits, et_logits, rc_logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def validate(model, loader, dice_metric, device):
    """Per-slice validation — used for fast in-training monitoring."""
    model.eval()
    dice_metric.reset()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].long().to(device)

            wt_logits, tc_logits, et_logits, rc_logits = model(images)

            preds = combine_hierarchical_predictions(wt_logits, tc_logits, et_logits, rc_logits)
            preds_onehot  = one_hot(preds,          num_classes=5)  # (B, 5, H, W)
            labels_onehot = one_hot(labels.long(),  num_classes=5)

            # Evaluate foreground classes only (channels 1-4: NETC, SNFH, ET, RC)
            dice_metric(preds_onehot[:, 1:], labels_onehot[:, 1:])

    per_class = dice_metric.aggregate()     # shape (4,)
    return per_class.mean().item(), per_class.tolist()


def main():
    parser = argparse.ArgumentParser(description="Hierarchical 2-D BRaTS Segmentation Training")
    parser.add_argument("--config", default="config_hierarchical_2d.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--synthetic_data_dir", default=None,
                        help="Directory of GAN-generated cases to add to training (optional)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    set_determinism(seed=config["seed"])

    device = get_device()
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    logger = setup_logging(config["output_dir"], name="train_hierarchical")
    writer = SummaryWriter(log_dir=os.path.join(config["output_dir"], "tb_logs"))

    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")

    train_loader, val_loader = get_dataloaders_2d(config, synthetic_data_dir=args.synthetic_data_dir)
    logger.info(f"Train cases: {len(train_loader.dataset)}, Val cases: {len(val_loader.dataset)}")

    model = build_model(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    loss_fn = HierarchicalLoss(
        lambda_tc=config["loss"].get("lambda_tc", 1.0),
        lambda_et=config["loss"].get("lambda_et", 1.0),
        lambda_rc=config["loss"].get("lambda_rc", 2.0),
        lambda_hausdorff=config["loss"].get("lambda_hausdorff", 0.0),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    max_epochs = config["training"]["max_epochs"]
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config["scheduler"]["warmup_epochs"],
        max_epochs=max_epochs,
    )
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", ignore_empty=True)

    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        start_epoch, best_dice = load_checkpoint(args.resume, model, optimizer, scheduler)
        logger.info(f"Resumed from epoch {start_epoch}, best_dice={best_dice:.4f}")

    logger.info("Starting hierarchical training...")
    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        scheduler.step()
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch [{epoch+1}/{max_epochs}] loss={avg_loss:.4f} lr={lr:.2e} time={elapsed:.1f}s")
        writer.add_scalar("train/loss", avg_loss, epoch + 1)
        writer.add_scalar("train/lr", lr, epoch + 1)

        if (epoch + 1) % config["training"]["val_interval"] == 0:
            logger.info("Running validation...")
            mean_dice, per_class = validate(model, val_loader, dice_metric, device)
            class_str = " | ".join(f"{n}={d:.4f}" for n, d in zip(CLASS_NAMES, per_class))
            logger.info(f"  Val mean Dice={mean_dice:.4f} | {class_str}")
            writer.add_scalar("val/mean_dice", mean_dice, epoch + 1)
            for name, d in zip(CLASS_NAMES, per_class):
                writer.add_scalar(f"val/dice_{name}", d, epoch + 1)

            if mean_dice > best_dice:
                best_dice = mean_dice
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, best_dice,
                    os.path.join(config["checkpoint_dir"], "best_model_hierarchical.pth"),
                )
                logger.info(f"  New best model saved (Dice={best_dice:.4f})")

        if (epoch + 1) % 50 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, best_dice,
                os.path.join(config["checkpoint_dir"], f"checkpoint_hierarchical_epoch_{epoch+1}.pth"),
            )

    writer.close()
    logger.info(f"Training complete. Best mean Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
