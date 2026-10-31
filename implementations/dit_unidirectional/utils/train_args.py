import argparse


class ArgsConfig:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Train a conditional Diffusion Transformer on video/time-series data",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        self.parser = parser
        self.get_train_args(parser)

        ns = parser.parse_args()

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
            choices=["ce-rm", "ce-rpui", "ns-pwc", "2d-reacdiff", "2d-swe"],
        )

        parser.add_argument("--dataset_root", type=str,
            default=None,
            help="Root directory used to resolve dataset paths.",
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
        
        # diffusion related hyperparameters
        parser.add_argument("--samp_diff_steps", type=int,
            default=100, 
            help="Reverse diffusion sampling steps.",)

        parser.add_argument("--num_diffusion_steps", type=int,
            default=1000, 
            help="Number of training diffusion timesteps",
        )

        parser.add_argument("--beta_start", type=float, default=1e-4,
            help="Start value for the linear DDPM beta schedule",
        )

        parser.add_argument("--beta_end", type=float, default=2e-2,
            help="End value for the linear DDPM beta schedule",
        )

        parser.add_argument("--ddim_eta", type=float, default=0.0,
            help="DDIM stochasticity during validation sampling; 0 is deterministic given initial noise",
        )

        parser.add_argument("--time_emb_dim", type=int, default=512,
            help="Dimension of time embedding",
        )

        parser.add_argument("--dit_arch", type=str, default="legacy",
            choices=["legacy", "adaln_zero"],
            help=(
                "DiT conditioning architecture. legacy keeps the previous "
                "input-only conditioning; adaln_zero conditions every block."
            ),
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
