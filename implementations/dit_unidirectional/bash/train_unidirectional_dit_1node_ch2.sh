#!/usr/bin/env bash
set -Eeuo pipefail

##### USER-TUNABLE DEFAULTS ####################################################
: "${PROJECT_DIR:=/lustre/scratch5/exempt/artimis/mrautela/my_projects/pde-dit}"
: "${DATASET_ROOT:=/lustre/scratch5/exempt/artimis/mrautela/my_projects/pde-dit}"
: "${CONDA_SH:=/users/mrautela/miniconda3/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=/lustre/scratch5/exempt/artimis/mrautela/.conda/envs/pytorch_py312_env}"
: "${CKPT_DIR:=${PROJECT_DIR}/models}"
: "${TRAIN_PY:=train_dit_unidirectional.py}"

# Chicoma: one node, four GPUs.
: "${ACCOUNT:=y26_artimis-test_g}"
: "${PARTITION:=gpu}"
: "${NNODES:=1}"
: "${GPUS:=4}"
: "${CPUS_PER_TASK:=32}"
: "${WALLTIME:=16:00:00}"
: "${CONSTRAINT:=gpu80}"

# 0 keeps requesting allocations until training finishes.
: "${MAX_ALLOCATIONS:=0}"

# Branch scale used for the nominal ~10M setting in the existing model_config.
: "${MODEL_SIZE:=s}"
: "${DATASET:=ce-rm}"
: "${EPOCHS:=2000}"
: "${BATCH_SIZE:=256}"

# Match the current adaLN-Zero branch.
: "${NUM_DIFFUSION_STEPS:=1000}"
: "${SAMP_DIFF_STEPS:=100}"
: "${PATCH_SIZE_H:=8}"
: "${PATCH_SIZE_W:=8}"

: "${RUN_NAME:=dit-adaln_zero-${MODEL_SIZE}_${DATASET}_forward_unidirectional}"

# ALLOW_FRESH=0 prevents an accidental restart from epoch 1.
: "${ALLOW_FRESH:=0}"
# DRY_RUN=1 prints the resolved Python command and exits before Slurm.
: "${DRY_RUN:=0}"

if (( NNODES != 1 )); then
  echo "[ERROR] Current trainer uses DataParallel. Set NNODES=1."
  exit 1
fi
if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "[ERROR] Project directory does not exist: ${PROJECT_DIR}"
  exit 1
fi
if [[ ! -f "${PROJECT_DIR}/${TRAIN_PY}" ]]; then
  echo "[ERROR] Training script does not exist: ${PROJECT_DIR}/${TRAIN_PY}"
  exit 1
fi

ARGS=(
  --dataset_root "${DATASET_ROOT}"
  --parallel
  --ms "${MODEL_SIZE}"
  --ds "${DATASET}"
  --epochs "${EPOCHS}"
  --bs "${BATCH_SIZE}"
  --dit_arch adaln_zero
  --num_diffusion_steps "${NUM_DIFFUSION_STEPS}"
  --samp_diff_steps "${SAMP_DIFF_STEPS}"
  --patch_size "${PATCH_SIZE_H}" "${PATCH_SIZE_W}"
)

build_python_command() {
  local run_dir latest_ckpt
  run_dir="${CKPT_DIR}/${RUN_NAME}"
  latest_ckpt="$(
    ls -t "${run_dir}"/dit-adaln_zero-"${MODEL_SIZE}"_"${DATASET}"_forward_unidirectional*.pth 2>/dev/null \
      | head -n 1 || true
  )"

  if [[ -n "${latest_ckpt}" && -s "${latest_ckpt}" ]]; then
    echo "[INFO] Resuming from: ${latest_ckpt}"
    CKPT_ARGS=(--resume --ckpt "${latest_ckpt}")
  elif [[ "${ALLOW_FRESH}" == "1" ]]; then
    echo "[WARN] No checkpoint found; ALLOW_FRESH=1, starting fresh."
    CKPT_ARGS=()
  else
    echo "[ERROR] No checkpoint found in ${run_dir} and ALLOW_FRESH=0."
    echo "[ERROR] For a new run: ALLOW_FRESH=1 bash $0"
    return 1
  fi

  printf -v PYTHON_CMD '%q ' \
    python -u "${PROJECT_DIR}/${TRAIN_PY}" "${ARGS[@]}" "${CKPT_ARGS[@]}"
}

echo "[INFO] PROJECT_DIR=${PROJECT_DIR}"
echo "[INFO] DATASET_ROOT=${DATASET_ROOT}"
echo "[INFO] CONDA_SH=${CONDA_SH}"
echo "[INFO] CONDA_ENV=${CONDA_ENV}"
echo "[INFO] MODEL_SIZE=${MODEL_SIZE}"
echo "[INFO] DATASET=${DATASET}"
echo "[INFO] DIT_ARCH=adaln_zero"
echo "[INFO] NUM_DIFFUSION_STEPS=${NUM_DIFFUSION_STEPS}"
echo "[INFO] SAMP_DIFF_STEPS=${SAMP_DIFF_STEPS}"
echo "[INFO] RUN_NAME=${RUN_NAME}"

if [[ "${DRY_RUN}" == "1" ]]; then
  build_python_command
  echo "[INFO] Python command:"
  echo "${PYTHON_CMD}"
  echo "[INFO] DRY_RUN=1; Slurm allocation not requested."
  exit 0
fi

# Use the known Chicoma conda initialization path explicitly.
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERROR] conda.sh not found at: ${CONDA_SH}"
  exit 1
fi

SALLOC_OPTS=(
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --nodes="${NNODES}"
  --ntasks=1
  --ntasks-per-node=1
  --gpus-per-node="${GPUS}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --time="${WALLTIME}"
  --constraint="${CONSTRAINT}"
  --job-name="${RUN_NAME}"
)

trap 'echo; echo "[INFO] Caught interrupt; exiting."; exit 130' INT TERM

ALLOC_IDX=0
while (( MAX_ALLOCATIONS == 0 || ALLOC_IDX < MAX_ALLOCATIONS )); do
  ALLOC_IDX=$((ALLOC_IDX + 1))
  echo "[INFO] Starting Chicoma allocation window ${ALLOC_IDX}"

  # Re-scan here so a new checkpoint from the previous allocation is used.
  build_python_command
  echo "[INFO] Python command:"
  echo "${PYTHON_CMD}"

  ALLOCATION_BODY=$(cat <<INNER_EOF
set -Eeuo pipefail
source $(printf '%q' "${CONDA_SH}")
conda activate $(printf '%q' "${CONDA_ENV}")
cd $(printf '%q' "${PROJECT_DIR}")

export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-${CPUS_PER_TASK}}"
export MKL_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-${CPUS_PER_TASK}}"

echo "Job ID: \${SLURM_JOB_ID}"
echo "Node list: \${SLURM_JOB_NODELIST}"
echo "CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: \$(command -v python)"
python --version

srun --export=ALL \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=$(printf '%q' "${CPUS_PER_TASK}") \
  --kill-on-bad-exit=1 \
  --cpu-bind=cores \
  ${PYTHON_CMD}
INNER_EOF
)

  if salloc "${SALLOC_OPTS[@]}" bash -lc "${ALLOCATION_BODY}"; then
    echo "[INFO] Training finished successfully at $(date)."
    break
  else
    rc=$?
    echo "[WARN] Allocation/training ended with exit code ${rc} at $(date)."
    if (( MAX_ALLOCATIONS == 0 || ALLOC_IDX < MAX_ALLOCATIONS )); then
      echo "[INFO] Waiting 60 seconds before requesting the next allocation."
      sleep 60
    fi
  fi
done
