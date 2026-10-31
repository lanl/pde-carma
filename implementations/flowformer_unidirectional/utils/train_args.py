import argparse


class ArgsConfig:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Train FlowFormer on video/time-series data",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        self.parser = parser
        self.get_train_args(parser)

        ns = parser.parse_args()
        ns.patch_size = tuple(ns.patch_size)

        for key, val in vars(ns).items():
            setattr(self, key, val)

    def get_train_args(self, parser: argparse.ArgumentParser):
        
        # device related
        parser.add_argument("--device_idx", type=int, default=0)
        parser.add_argument("--parallel", action="store_true", 
                            help="Use DataParallel if multiple GPUs are available"
        )
        
        # patching related
        parser.add_argument("--patch_size", type=int,
            nargs=2, default=(8, 8), metavar=("PH", "PW"),
            help="Patch size for ViT as two integers: --patch_size 8 8",
        )

        # model and dataset choices
        parser.add_argument("--ms", type=str, default="ti",
            choices=["ti", "mi", "s", "m", "b", "l", "xl", "xxl"],
        )

        parser.add_argument("--ds", type=str, default="ce-rm",
            choices=["ce-rm", "ce-rpui", "ns-pwc", "2d-reacdiff", "2d-swe", "fns-kf"],
        )

        parser.add_argument(
            "--dataset_root",
            type=str,
            default=None,
            help="Optional root containing the datasets/ directory",
        )

        # direction-specific training
        parser.add_argument(
            "--direction",
            type=str,
            default="forward",
            choices=["forward", "backward"],
            help="Train a direction-specific model: forward or backward",
        )

        # training hyperparameters
        parser.add_argument("--epochs", type=int, default=2000,
            help="Number of training epochs",
        )

        parser.add_argument("--bs", type=int, default=256,
            help="Batch size",
        )

        parser.add_argument("--lr", type=float, default=1e-4,
            help="Learning rate",
        )

        parser.add_argument("--num_workers", type=int, default=0)
        
        # flow related hyperparameters
        parser.add_argument("--num_ode_steps", type=int, default=5,
            help="Number of ODE steps for sampling",
        )

        parser.add_argument("--time_emb_dim", type=int, default=512,
            help="Dimension of time embedding",
        )

        # resume related
        parser.add_argument("--resume", action="store_true",
            help="Whether to resume training from checkpoint specified by --ckpt",
        )

        parser.add_argument("--ckpt", type=str, default=None,
            help="Checkpoint filename or path to load when using --resume",
        )

        parser.add_argument(
            "--tag",
            type=str,
            default="",
            help="Optional experiment tag",
        )