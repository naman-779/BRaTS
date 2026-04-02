import argparse
import os
import time

import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter

from src.data import get_dataloaders
from src.losses import build_loss
from src.model import build_model
from src.scheduler import WarmupCosineScheduler
from src.utils import (
    get_device,
    load_checkpoint,
    load_config,
    save_checkpoint,
    set_seed,
    setup_logging,
)

CLASS_NAMES = ["NETC", "SNFH", "ET", "RC"]


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, amp_enabled):
    model.train()
    epoch_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled and device.type == "cuda"):
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        if amp_enabled and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def validate(model, loader, dice_metric, device, roi_size, sw_batch_size, overlap, mode, amp_enabled):
    model.eval()
    dice_metric.reset()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=amp_enabled and device.type == "cuda"):
                outputs = sliding_window_inference(
                    images,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode=mode,
                )

            # argmax -> one-hot for metric computation
            preds = torch.argmax(outputs, dim=1, keepdim=True)
            preds_onehot = one_hot(preds, num_classes=5)
            labels_onehot = one_hot(labels.long(), num_classes=5)

            # Compute dice for foreground classes only (channels 1-4)
            dice_metric(preds_onehot[:, 1:], labels_onehot[:, 1:])

    per_class = dice_metric.aggregate()  # shape: (4,)
    mean_dice = per_class.mean().item()
    per_class_list = per_class.tolist()
    return mean_dice, per_class_list


def main():
    parser = argparse.ArgumentParser(description="BRaTS 2024 Baseline Training")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    set_determinism(seed=config["seed"])

    device = get_device()
    output_dir = config["output_dir"]
    checkpoint_dir = config["checkpoint_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = setup_logging(output_dir)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb_logs"))

    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")

    # Data
    logger.info("Building dataloaders...")
    train_loader, val_loader = get_dataloaders(config)
    logger.info(f"Train: {len(train_loader.dataset)} cases, Val: {len(val_loader.dataset)} cases")

    # Model
    model = build_model(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Loss, optimizer, scheduler
    loss_fn = build_loss(config)
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
    amp_enabled = config["training"]["amp"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")

    # Metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", ignore_empty=True)

    # Resume
    start_epoch = 0
    best_dice = 0.0
    if args.resume:
        start_epoch, best_dice = load_checkpoint(args.resume, model, optimizer, scheduler)
        logger.info(f"Resumed from epoch {start_epoch}, best_dice={best_dice:.4f}")

    # Inference config
    roi_size = tuple(config["inference"]["roi_size"])
    sw_batch_size = config["inference"]["sw_batch_size"]
    overlap = config["inference"]["overlap"]
    sw_mode = config["inference"]["mode"]
    val_interval = config["training"]["val_interval"]

    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()
        avg_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, amp_enabled
        )
        scheduler.step()
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch [{epoch + 1}/{max_epochs}] loss={avg_loss:.4f} lr={lr:.2e} time={elapsed:.1f}s"
        )
        writer.add_scalar("train/loss", avg_loss, epoch + 1)
        writer.add_scalar("train/lr", lr, epoch + 1)

        # Validation
        if (epoch + 1) % val_interval == 0:
            logger.info("Running validation...")
            mean_dice, per_class = validate(
                model, val_loader, dice_metric, device,
                roi_size, sw_batch_size, overlap, sw_mode, amp_enabled,
            )

            class_str = " | ".join(
                f"{name}={d:.4f}" for name, d in zip(CLASS_NAMES, per_class)
            )
            logger.info(f"  Val mean Dice={mean_dice:.4f} | {class_str}")

            writer.add_scalar("val/mean_dice", mean_dice, epoch + 1)
            for name, d in zip(CLASS_NAMES, per_class):
                writer.add_scalar(f"val/dice_{name}", d, epoch + 1)

            if mean_dice > best_dice:
                best_dice = mean_dice
                save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, best_dice,
                    os.path.join(checkpoint_dir, "best_model.pth"),
                )
                logger.info(f"  New best model saved (Dice={best_dice:.4f})")

        # Save latest checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, best_dice,
                os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth"),
            )

    writer.close()
    logger.info(f"Training complete. Best mean Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
