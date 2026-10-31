# Reproducibility

The repository preserves source scripts rather than rewriting them into a new experiment framework. The unified launcher can execute an allowlisted original script while maintaining its implementation working directory:

```bash
python scripts/run.py reproduce VARIANT SOURCE_RELATIVE_SCRIPT -- [original script arguments]
```

List registered source scripts with:

```bash
python scripts/run.py list --reproduction
```

Dry-run the exact child command before an expensive experiment:

```bash
python scripts/run.py --dry-run reproduce dit-bidirectional \
  scripts/generate_tensors_plot_error_bar_dit.py -- --help
```

## Source-backed experiment families

### DiT

- Uni-DiT: `scripts/evaluate_unidirectional_dit_multirun.py` is the designated forward-only evaluator.
- Bi-DiT: table, tensor/error-bar, multirun, cyclic-distribution, and rollout-plot scripts remain in the snapshot.
- Uni-DiT also contains cyclic table/tensor scripts inherited from the bidirectional workflow. Because its designated trainer is forward-only, they remain as needs-review provenance and are deliberately not allowlisted by the external launcher.

### FlowFormer

- Uni-FlowFormer: `scripts/generate_unifwd_table3.py` evaluates direction-specific checkpoints. Its inherited `plot_carma_rollouts.py` expects a bidirectional control-MLP checkpoint, so the external launcher deliberately does not allowlist it for Uni-FlowFormer.
- Bi-FlowFormer: table, tensor/error-bar, multirun, cyclic-distribution, rollout, and ODE-sweep scripts remain available.
- Adaptive, greedy, strong-greedy, and oracle scripts remain exploratory. Oracle selection uses ground truth and is evaluation-only.

The Bi-FlowFormer snapshot contains byte-identical root and `scripts/` copies of the basic ODE sweep. Its two resume scripts are not identical: one activates CE-RPUI/NS-PwC runs and the other activates FNS-KF runs. Both remain intact; the launcher must not silently choose one as equivalent to the other.

## Recorded paper settings

The migration specification records these flow-matching ODE settings for paper reproduction:

| Dataset | ODE steps |
|---|---:|
| CE-RM | 5 |
| CE-RPUI | 5 |
| NS-PwC | 100 |
| FNS-KF | 100 |
| NS-SL | 50 |

Candidate-budget ablations may use `K = 4, 8, 16`. These are experiment settings, not replacements for source defaults. The source ODE sweep also examines `1, 5, 20, 50, 100`.

The Uni-FlowFormer Table-3 material records `K=8`, 10 realizations, base seed 2027, and five ODE steps. Preserve its script-specific seed handling rather than applying it globally.

## Numerical invariants

The following remain implementation-local and unchanged:

- direction pairing/control semantics;
- channel selection, trajectory splits, and normalization;
- DiT beta schedule, diffusion training timesteps, DDIM reverse sampling, and eta;
- flow interpolation, target velocity, loss, and explicit-Euler integration;
- candidate indexing and random-number consumption;
- endpoint/path cyclic reductions and metric formulas;
- checkpoint dictionaries and state-dict keys.

The Uni-DiT defaults (`1000` training diffusion steps, `100` sampling steps) and Bi-DiT defaults (`100`, `1000`) differ materially and are retained pending author verification.

## Seeds and candidate pools

The primary train/inference entry points do not set a global seed. Some multirun scripts do. Record, at minimum:

- Python, NumPy, and PyTorch seeds/states;
- software versions and accelerator type;
- model/dataset/direction identifiers;
- checkpoint hash and saved arguments;
- sampling or ODE steps and DDIM eta;
- `K`, candidate-pool construction, and selection score;
- normalization/split provenance.

Do not assume that running multiple `K` values in the compatibility evaluator produces nested candidate pools; it resamples per `K`.

## Reproduction status

The migration validates structure, hashes, wrapper resolution, and lightweight syntax where feasible. It does not reproduce a paper table or figure: datasets, trained checkpoints, result tensors, GPUs, and the complete ML environment were not supplied. Any future reproduction report must distinguish completed experiments from command templates.

