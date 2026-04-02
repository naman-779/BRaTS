import torch
import torch.nn as nn


def generator_loss(d_fake_output, real_scan, fake_scan, lambda1, lambda2):
    """Generator loss from the paper (Eq. 3).

    L_G = -lambda1 * E[log D(G(z|y))] + lambda2 * ||x - G(z|y)||_MAE

    Args:
        d_fake_output: discriminator output on fake scan
        real_scan: ground truth clean scan (B, 4, D, H, W)
        fake_scan: generator output (B, 4, D, H, W)
        lambda1: adversarial loss weight
        lambda2: MAE reconstruction loss weight
    """
    # Adversarial loss: -E[log(D(fake))]
    adv_loss = -torch.mean(torch.log(d_fake_output + 1e-8))

    # MAE reconstruction loss
    mae_loss = nn.functional.l1_loss(fake_scan, real_scan)

    return lambda1 * adv_loss + lambda2 * mae_loss, adv_loss.item(), mae_loss.item()


def discriminator_loss(d_real_output, d_fake_output):
    """Discriminator loss from the paper (Eq. 4).

    L_D = E[log D(G(z|y))] - E[log D(x|y)]
         = -E[log D(x|y)] - E[log(1 - D(G(z|y)))]  (standard GAN form)

    Args:
        d_real_output: discriminator output on real scan
        d_fake_output: discriminator output on fake scan
    """
    real_loss = -torch.mean(torch.log(d_real_output + 1e-8))
    fake_loss = -torch.mean(torch.log(1.0 - d_fake_output + 1e-8))
    return real_loss + fake_loss
