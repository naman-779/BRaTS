import math
import torch


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by cosine annealing to zero."""

    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.max_epochs - self.warmup_epochs
        )
        return [
            base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]
