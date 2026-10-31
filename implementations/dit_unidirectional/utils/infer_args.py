import argparse


class InferArgsConfig:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Inference for conditional DiT",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        self.parser = parser
        self.get_infer_args(parser)

        ns = parser.parse_args()

        for key, val in vars(ns).items():
            setattr(self, key, val)

    def get_infer_args(self, parser: argparse.ArgumentParser):

        parser.add_argument("--ckpt", type=str, required=True)
        parser.add_argument("--device_idx", type=int, default=0)
        
        parser.add_argument("--dataset_root", type=str,
            default=None, help="Root directory.",
        )

        parser.add_argument(
            "--samp_diff_steps",
            type=int,
            default=None,
            help="Override checkpoint reverse diffusion sampling steps.",
        )
        parser.add_argument(
            "--ddim_eta",
            type=float,
            default=None,
            help="Override checkpoint DDIM eta during sampling",
        )
        parser.add_argument("--num_runs", type=int, default=1)

        parser.add_argument("--num_samples", type=int, default=100)
        parser.add_argument("--bs", type=int, default=4)

        parser.add_argument("--viz_frame_nsp", type=int, default=0)
        parser.add_argument("--num_sampling_frame", type=int, default=5)

        parser.add_argument("--max_T_rollout", type=int, default=20)
        parser.add_argument("--num_traj_rollout", type=int, default=1)

        parser.add_argument(
            "--cyclic_runs",
            type=int,
            nargs="+",
            default=[1],
            help="Number of stochastic cyclic candidates, e.g. --cyclic_runs 16 64 128",
        )

        
