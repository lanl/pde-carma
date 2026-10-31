import math
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)

class LRSched:
    @staticmethod
    def plateau(optimizer, args):
        """
        ReduceLROnPlateau on val loss.
        """
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,          
            patience=5, 
            threshold=1e-5,
        )
        return scheduler

    @staticmethod
    def warmup_cosine(optimizer, args):
        """
        Linear warmup for a few epochs, then cosine decay.
        """
        total_epochs  = args.epochs
        warm_epochs = 5
        warmup_epochs = max(0, warm_epochs)  
        cosine_epochs = max(0, total_epochs - warmup_epochs)

        # extra args with safe defaults
        min_lr = 1e-9
        warm_start_factor = min_lr / args.lr # (1e-9/1e-4 = 1e-5)

        # linear warmup: start_factor * base_lr --> 1.0 * base_lr
        sched_warmup = LinearLR(
            optimizer,
            start_factor=warm_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        # cosine decay from base_lr to min_lr
        sched_cosine = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=min_lr,
        )

        # chain them together
        scheduler = SequentialLR(
            optimizer,
            schedulers=[sched_warmup, sched_cosine],
            milestones=[warmup_epochs],  
        )
        return scheduler
