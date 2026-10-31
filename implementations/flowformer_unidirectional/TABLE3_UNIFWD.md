# Table 3: Uni-Fwd runs

Table 3 uses the forward-only FlowFormer at the paper's ~10M scale. In this
codebase that scale is `--ms s` (the next size, `--ms m`, is the ~25M model).

## 1. Train on Chicoma

The Chicoma launcher is `bash/train_1node_dp_ch.sh`. For a new run:

```bash
ALLOW_FRESH=1 MODEL_SIZE=s DIRECTION=forward DATASET=ce-rm \
  bash bash/train_1node_dp_ch.sh
```

Repeat with:

- `ce-rm`
- `ce-rpui`
- `ns-pwc`
- `2d-reacdiff`
- `2d-swe`

After the first allocation, leave `ALLOW_FRESH=0` (the default). The launcher
finds the highest numbered checkpoint and resumes it.

If the datasets live somewhere else, set `DATASET_ROOT=/path/to/pde-flowformer`.
It must contain a `datasets/` directory with the same layout as the current
bidirectional branch.

## 2. Evaluate Uni-Fwd with the Table-3 protocol

Example for CE-RM:

```bash
python scripts/generate_unifwd_table3.py \
  --ckpt models/flowformer-s_ce-rm_forward_2000.pth \
  --dataset datasets/vids_test_ce-rm_forward.npy \
  --K 8 \
  --R 10 \
  --base_seed 2027 \
  --num_ode_steps 5 \
  --max_T_rollout 20 \
  --traj_batch_size 4
```

Change both the checkpoint and test filename for each dataset. The script
reports the mean and sample standard deviation across the `R=10`
realization-level Forward MSE values, and saves the per-run values plus the
full `(R,K,N_test)` candidate-error tensor.

The current bidirectional multi-run evaluator defaults to 5 ODE steps. If the
runs used to generate the paper's Bi-Fwd numbers explicitly overrode that
value, use the same value here instead.
