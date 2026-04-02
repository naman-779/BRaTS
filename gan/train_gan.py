import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from gan.dataset import GanTrainDataset
from gan.discriminator import PatchDiscriminator3D
from gan.losses import generator_loss, discriminator_loss
from src.utils import get_device, load_config, set_seed, setup_logging

from monai.networks.nets import SwinUNETR


def build_generator(cfg):
    gcfg = cfg["gan"]["generator"]
    return SwinUNETR(
        in_channels=gcfg["in_channels"],
        out_channels=gcfg["out_channels"],
        feature_size=gcfg["feature_size"],
        spatial_dims=gcfg.get("spatial_dims", 3),
    )


def build_discriminator(cfg):
    dcfg = cfg["gan"]["discriminator"]
    return PatchDiscriminator3D(
        in_channels=dcfg["in_channels"],
        base_filters=dcfg["base_filters"],
    )


def train_phase1(generator, discriminator, dataloader, opt_g, opt_d, device, cfg, logger):
    """Phase 1: fixed lambda1=1, lambda2=5 for N iterations."""
    tcfg = cfg["gan"]["training"]
    max_iters = tcfg["phase1_iterations"]
    lambda1 = tcfg["lambda1"]
    lambda2 = tcfg["lambda2_start"]
    g_steps = tcfg["g_steps"]

    generator.train()
    discriminator.train()

    iteration = 0
    while iteration < max_iters:
        for batch in dataloader:
            if iteration >= max_iters:
                break

            noisy_input = batch["noisy_input"].to(device)
            real_scan = batch["real_scan"].to(device)
            label = batch["label"].to(device)
            # Normalize label for discriminator conditioning
            label_norm = label / max(label.max().item(), 1.0)

            # --- Train Discriminator ---
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_scan = generator(noisy_input)
            # Concat scan + label for discriminator
            d_real_input = torch.cat([real_scan, label_norm], dim=1)
            d_fake_input = torch.cat([fake_scan, label_norm], dim=1)
            d_real_out = discriminator(d_real_input)
            d_fake_out = discriminator(d_fake_input)
            d_loss = discriminator_loss(d_real_out, d_fake_out)
            d_loss.backward()
            opt_d.step()

            # --- Train Generator (g_steps times) ---
            for _ in range(g_steps):
                opt_g.zero_grad(set_to_none=True)
                fake_scan = generator(noisy_input)
                d_fake_input = torch.cat([fake_scan, label_norm], dim=1)
                d_fake_out = discriminator(d_fake_input)
                g_loss, adv_val, mae_val = generator_loss(
                    d_fake_out, real_scan, fake_scan, lambda1, lambda2
                )
                g_loss.backward()
                opt_g.step()

            iteration += 1
            if iteration % 50 == 0 or iteration == 1:
                logger.info(
                    f"Phase1 [{iteration}/{max_iters}] "
                    f"G_loss={g_loss.item():.4f} (adv={adv_val:.4f} mae={mae_val:.4f}) "
                    f"D_loss={d_loss.item():.4f}"
                )


def train_phase2(generator, discriminator, dataloader, opt_g, opt_d, device, cfg, logger):
    """Phase 2: linearly increase lambda2 from start to end over N epochs."""
    tcfg = cfg["gan"]["training"]
    max_epochs = tcfg["phase2_epochs"]
    lambda2_start = tcfg["lambda2_start"]
    lambda2_end = tcfg["lambda2_end"]
    g_steps = tcfg["g_steps"]

    generator.train()
    discriminator.train()

    for epoch in range(max_epochs):
        # Linear ramp: lambda2 from start to end
        lambda2 = lambda2_start + (lambda2_end - lambda2_start) * epoch / max(max_epochs - 1, 1)
        lambda1 = 1.0 / lambda2

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            noisy_input = batch["noisy_input"].to(device)
            real_scan = batch["real_scan"].to(device)
            label = batch["label"].to(device)
            label_norm = label / max(label.max().item(), 1.0)

            # --- Discriminator ---
            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_scan = generator(noisy_input)
            d_real_input = torch.cat([real_scan, label_norm], dim=1)
            d_fake_input = torch.cat([fake_scan, label_norm], dim=1)
            d_real_out = discriminator(d_real_input)
            d_fake_out = discriminator(d_fake_input)
            d_loss = discriminator_loss(d_real_out, d_fake_out)
            d_loss.backward()
            opt_d.step()

            # --- Generator ---
            for _ in range(g_steps):
                opt_g.zero_grad(set_to_none=True)
                fake_scan = generator(noisy_input)
                d_fake_input = torch.cat([fake_scan, label_norm], dim=1)
                d_fake_out = discriminator(d_fake_input)
                g_loss, adv_val, mae_val = generator_loss(
                    d_fake_out, real_scan, fake_scan, lambda1, lambda2
                )
                g_loss.backward()
                opt_g.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            n_batches += 1

        avg_g = epoch_g_loss / max(n_batches, 1)
        avg_d = epoch_d_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Phase2 Epoch [{epoch + 1}/{max_epochs}] "
                f"G_loss={avg_g:.4f} D_loss={avg_d:.4f} "
                f"λ1={lambda1:.4f} λ2={lambda2:.1f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Train GliGAN")
    parser.add_argument("--config", type=str, default="config_gan.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device()

    ckpt_dir = config.get("checkpoint_dir", "./checkpoints/gan")
    output_dir = config.get("output_dir", "./runs/gan")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Device: {device}")

    # Dataset
    tcfg = config["gan"]["training"]
    crop_size = tuple(config["gan"]["generation"]["crop_size"])
    dataset = GanTrainDataset(
        data_root=config["data_root"],
        train_dir=config["train_dir"],
        modalities=config["modalities"],
        crop_size=crop_size,
        data_fraction=config.get("data_fraction", 1.0),
        seed=config.get("seed", 42),
    )
    loader = DataLoader(
        dataset,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    logger.info(f"GAN training dataset: {len(dataset)} cases")

    # Models
    generator = build_generator(config).to(device)
    discriminator = build_discriminator(config).to(device)
    g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    logger.info(f"Generator parameters: {g_params:,}")
    logger.info(f"Discriminator parameters: {d_params:,}")

    # Optimizers
    lr = tcfg["lr"]
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Phase 1
    logger.info("=== Phase 1: Fixed lambdas ===")
    t0 = time.time()
    train_phase1(generator, discriminator, loader, opt_g, opt_d, device, config, logger)
    logger.info(f"Phase 1 complete in {time.time() - t0:.0f}s")

    # Save phase 1 checkpoint
    torch.save(generator.state_dict(), os.path.join(ckpt_dir, "generator_phase1.pth"))
    torch.save(discriminator.state_dict(), os.path.join(ckpt_dir, "discriminator_phase1.pth"))

    # Phase 2
    logger.info("=== Phase 2: Lambda ramp ===")
    t0 = time.time()
    train_phase2(generator, discriminator, loader, opt_g, opt_d, device, config, logger)
    logger.info(f"Phase 2 complete in {time.time() - t0:.0f}s")

    # Save final checkpoint
    torch.save(generator.state_dict(), os.path.join(ckpt_dir, "generator_best.pth"))
    torch.save(discriminator.state_dict(), os.path.join(ckpt_dir, "discriminator_best.pth"))
    logger.info(f"Checkpoints saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
