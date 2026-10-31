#!/usr/bin/env bash
set -Eeuo pipefail

##### USER-TUNABLE DEFAULTS ####################################################
: "${PROJECT_DIR:=/lustre/scratch5/exempt/artimis/mrautela/my_projects/pde-dit}"
: "${CONDA_ENV:=/lustre/scratch5/exempt/artimis/mrautela/.conda/envs/pytorch_py312_env}"
: "${CKPT_DIR:=${PROJECT_DIR}/models}"
: "${TRAIN_PY:=train_dit.py}"

# Chicoma Slurm resources: one full node with four GPUs.
: "${ACCOUNT:=y26_artimis-fmod_g}"
: "${PARTITION:=gpu}"
: "${NNODES:=1}"
: "${GPUS:=4}"
: "${CPUS_PER_TASK:=32}"
: "${WALLTIME:=16:00:00}"
: "${CONSTRAINT:=gpu80}"

# MAX_ALLOCATIONS=0 means keep requesting new 16-hour allocations until
# training finishes. Set MAX_ALLOCATIONS=1 to request only one allocation.
: "${MAX_ALLOCATIONS:=0}"

# Training configuration.
: "${MODEL_SIZE:=s}"
: "${DATASET:=ce-rpui}"
: "${EPOCHS:=2000}"
: "${RUN_NAME:=dit-${MODEL_SIZE}_${DATASET}}"

# Safety switch:
#   ALLOW_FRESH=0: require an existing checkpoint.
#   ALLOW_FRESH=1: allow training to start from epoch 1.
: "${ALLOW_FRESH:=0}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "[ERROR] Project directory does not exist: ${PROJECT_DIR}"
  exit 1
fi

if [[ ! -f "${PROJECT_DIR}/${TRAIN_PY}" ]]; then
  echo "[ERROR] Training script does not exist: ${PROJECT_DIR}/${TRAIN_PY}"
  exit 1
fi

# Locate conda.sh so the named environment can be activated inside the
# allocated compute-node shell.
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

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERROR] conda.sh not found at: ${CONDA_SH}"
  exit 1
fi

# The dataset has already been downloaded, so --dataset_root is not passed.
ARGS=(
  --parallel
  --ms "${MODEL_SIZE}"
  --ds "${DATASET}"
  --epochs "${EPOCHS}"
)

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

echo "[INFO] Starting Chicoma allocation loop at $(date)"
echo "[INFO] PROJECT_DIR=${PROJECT_DIR}"
echo "[INFO] CONDA_ENV=${CONDA_ENV}"
echo "[INFO] CKPT_DIR=${CKPT_DIR}"
echo "[INFO] ACCOUNT=${ACCOUNT}"
echo "[INFO] PARTITION=${PARTITION}"
echo "[INFO] MODEL_SIZE=${MODEL_SIZE}"
echo "[INFO] DATASET=${DATASET}"
echo "[INFO] EPOCHS=${EPOCHS}"
echo "[INFO] RUN_NAME=${RUN_NAME}"
echo "[INFO] ALLOW_FRESH=${ALLOW_FRESH}"
echo "[INFO] Requesting ${NNODES} node, ${GPUS} GPUs, ${CPUS_PER_TASK} CPUs"
echo "[INFO] GPU constraint: ${CONSTRAINT}"

trap 'echo; echo "[INFO] Caught interrupt; exiting."; exit 130' INT TERM

ALLOC_IDX=0

while (( MAX_ALLOCATIONS == 0 || ALLOC_IDX < MAX_ALLOCATIONS )); do
  ALLOC_IDX=$((ALLOC_IDX + 1))

  if (( MAX_ALLOCATIONS == 0 )); then
    echo "[INFO] Starting allocation window ${ALLOC_IDX}/infinite"
  else
    echo "[INFO] Starting allocation window ${ALLOC_IDX}/${MAX_ALLOCATIONS}"
  fi

  RUN_DIR="${CKPT_DIR}/${RUN_NAME}"

  # Accept checkpoint names such as:
  #   dit-s_ce-rpui.pth
  #   dit-s_ce-rpui_574.pth
  LATEST_CKPT="$(
    ls -t "${RUN_DIR}"/dit-"${MODEL_SIZE}"_"${DATASET}"*.pth 2>/dev/null \
      | head -n 1 || true
  )"

  echo "[INFO] Looking for a checkpoint in: ${RUN_DIR}"

  if [[ -n "${LATEST_CKPT}" && -s "${LATEST_CKPT}" ]]; then
    echo "[INFO] Found checkpoint: ${LATEST_CKPT}"
    echo "[INFO] Training will resume from this checkpoint."
    CKPT_ARGS=(--resume --ckpt "${LATEST_CKPT}")
  else
    echo "[WARN] No checkpoint found in: ${RUN_DIR}"

    if [[ "${ALLOW_FRESH}" == "1" ]]; then
      echo "[WARN] ALLOW_FRESH=1; starting a new training run."
      CKPT_ARGS=()
    else
      echo "[ERROR] Refusing to start from epoch 1 because ALLOW_FRESH=0."
      echo "[ERROR] For a new run, execute:"
      echo "[ERROR] ALLOW_FRESH=1 bash $0"
      exit 1
    fi
  fi

  PYTHON_CMD="$(printf '%q ' python -u "${PROJECT_DIR}/${TRAIN_PY}" "${ARGS[@]}" "${CKPT_ARGS[@]}")"

  echo "[INFO] Training command:"
  echo "${PYTHON_CMD}"
  echo "[INFO] Requesting a ${WALLTIME} allocation from Chicoma..."

  ALLOCATION_BODY=$(cat <<INNER_EOF
set -Eeuo pipefail

source $(printf '%q' "${CONDA_SH}")
conda activate $(printf '%q' "${CONDA_ENV}")

cd $(printf '%q' "${PROJECT_DIR}")

export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-${CPUS_PER_TASK}}"
export MKL_NUM_THREADS="\${SLURM_CPUS_PER_TASK:-${CPUS_PER_TASK}}"

echo "========================================"
echo "Job ID: \${SLURM_JOB_ID}"
echo "Node list: \${SLURM_JOB_NODELIST}"
echo "CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-unset}"
echo "Repository: \$(pwd)"
echo "Conda environment: \${CONDA_DEFAULT_ENV:-unset}"
echo "Python: \$(command -v python)"
python --version
echo "========================================"

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

if (( MAX_ALLOCATIONS == 0 )); then
  echo "[INFO] Allocation loop ended at $(date)."
else
  echo "[INFO] Reached MAX_ALLOCATIONS=${MAX_ALLOCATIONS}. Exiting at $(date)."
fi