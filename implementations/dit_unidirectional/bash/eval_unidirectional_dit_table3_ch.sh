#!/usr/bin/env bash
set -Eeuo pipefail

##### USER-TUNABLE DEFAULTS ####################################################
: "${PROJECT_DIR:=/lustre/scratch5/exempt/artimis/mrautela/my_projects/pde-dit}"
: "${DATASET_ROOT:=/lustre/scratch5/exempt/artimis/mrautela/my_projects/pde-flowformer}"
: "${CONDA_ENV:=/lustre/scratch5/exempt/artimis/mrautela/.conda/envs/pytorch_py312_env}"
: "${EVAL_PY:=scripts/evaluate_unidirectional_dit_multirun.py}"

: "${ACCOUNT:=y26_artimis-fmod_g}"
: "${PARTITION:=gpu}"
: "${GPUS:=1}"
: "${CPUS_PER_TASK:=8}"
: "${WALLTIME:=16:00:00}"
: "${CONSTRAINT:=gpu80}"

: "${CKPT:=}"
: "${K:=8}"
: "${R:=10}"
: "${BASE_SEED:=2027}"
: "${SAMP_DIFF_STEPS:=100}"
: "${DDIM_ETA:=0.0}"
: "${MAX_T_ROLLOUT:=20}"
: "${TRAJ_BATCH_SIZE:=1}"
: "${OUT_DIR:=results/table3_unidirectional}"
: "${DRY_RUN:=0}"

if [[ -z "${CKPT}" ]]; then
  echo "[ERROR] Set CKPT to the forward-only checkpoint."
  exit 1
fi
if [[ ! -f "${PROJECT_DIR}/${EVAL_PY}" ]]; then
  echo "[ERROR] Evaluation script not found: ${PROJECT_DIR}/${EVAL_PY}"
  exit 1
fi

ARGS=(
  --ckpt "${CKPT}"
  --dataset_root "${DATASET_ROOT}"
  --K "${K}"
  --R "${R}"
  --base_seed "${BASE_SEED}"
  --samp_diff_steps "${SAMP_DIFF_STEPS}"
  --ddim_eta "${DDIM_ETA}"
  --max_T_rollout "${MAX_T_ROLLOUT}"
  --traj_batch_size "${TRAJ_BATCH_SIZE}"
  --out_dir "${OUT_DIR}"
)

printf -v PYTHON_CMD '%q ' python -u "${PROJECT_DIR}/${EVAL_PY}" "${ARGS[@]}"
echo "[INFO] Python command:"
echo "${PYTHON_CMD}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[INFO] DRY_RUN=1; Slurm allocation not requested."
  exit 0
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  CONDA_BASE="${HOME}/miniconda3"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  CONDA_BASE="${HOME}/anaconda3"
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  CONDA_BASE="${HOME}/miniforge3"
else
  echo "[ERROR] Could not locate conda.sh."
  exit 1
fi
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"

ALLOCATION_BODY=$(cat <<INNER_EOF
set -Eeuo pipefail
source $(printf '%q' "${CONDA_SH}")
conda activate $(printf '%q' "${CONDA_ENV}")
cd $(printf '%q' "${PROJECT_DIR}")
export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-${CPUS_PER_TASK}}"
export MKL_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-${CPUS_PER_TASK}}"

srun --export=ALL \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gpus-per-task=$(printf '%q' "${GPUS}") \
  --cpus-per-task=$(printf '%q' "${CPUS_PER_TASK}") \
  --kill-on-bad-exit=1 \
  --cpu-bind=cores \
  ${PYTHON_CMD}
INNER_EOF
)

salloc \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --gpus-per-node="${GPUS}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --time="${WALLTIME}" \
  --constraint="${CONSTRAINT}" \
  --job-name="dit-uni-table3" \
  bash -lc "${ALLOCATION_BODY}"
