#!/usr/bin/env bash
set -Eeuo pipefail

##### USER-TUNABLE DEFAULTS ####################################################
: "${PROJECT_DIR:=/lustre/scratch5/exempt/artimis/mrautela/my_projects/pde-flowformer}"
: "${DATASET_ROOT:=/lustre/scratch5/exempt/artimis/mrautela/my_projects/pde-dit}"
: "${CONDA_SH:=/users/mrautela/miniconda3/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=/lustre/scratch5/exempt/artimis/mrautela/.conda/envs/pytorch_py312_env}"
: "${CKPT_DIR:=${PROJECT_DIR}/models}"
: "${TRAIN_PY:=train_flowformer.py}"

# Chicoma: one full GPU node.
: "${ACCOUNT:=y26_artimis-test_g}" #y26_artimis-fmod_g
: "${PARTITION:=gpu}"
: "${NNODES:=1}"
: "${GPUS:=4}"
: "${CPUS_PER_TASK:=32}"
: "${WALLTIME:=16:00:00}"
: "${CONSTRAINT:=gpu80}"

# MAX_ALLOCATIONS=0 keeps requesting new allocations until training finishes.
: "${MAX_ALLOCATIONS:=0}"

# Table-3 target is the ~10M configuration. In model_config.py that is ms=s.
: "${MODEL_SIZE:=s}"
: "${DIRECTION:=forward}"
: "${DATASET:=ce-rm}"
: "${EPOCHS:=2000}"
: "${NUM_ODE_STEPS:=5}"
: "${BATCH_SIZE:=256}"
: "${RUN_NAME:=flowformer-${MODEL_SIZE}_${DATASET}_${DIRECTION}}"

# Current Table-3 datasets are all 128x128 and use 8x8 patches.
: "${PATCH_H:=8}"
: "${PATCH_W:=8}"

# ALLOW_FRESH=0 refuses to start from epoch 1 when no checkpoint is found.
: "${ALLOW_FRESH:=0}"

if (( NNODES != 1 )); then
  echo "[ERROR] train_flowformer.py uses DataParallel; set NNODES=1."
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

if [[ ! -d "${DATASET_ROOT}/datasets" ]]; then
  echo "[ERROR] Dataset directory does not exist: ${DATASET_ROOT}/datasets"
  exit 1
fi

# Chicoma conda initialization path used in the working interactive setup.
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[ERROR] conda.sh not found at: ${CONDA_SH}"
  exit 1
fi

ARGS=(
  --parallel
  --ms "${MODEL_SIZE}"
  --ds "${DATASET}"
  --direction "${DIRECTION}"
  --epochs "${EPOCHS}"
  --num_ode_steps "${NUM_ODE_STEPS}"
  --bs "${BATCH_SIZE}"
  --patch_size "${PATCH_H}" "${PATCH_W}"
  --dataset_root "${DATASET_ROOT}"
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

mkdir -p "${CKPT_DIR}"

echo "[INFO] Starting Chicoma allocation loop at $(date)"
echo "[INFO] PROJECT_DIR=${PROJECT_DIR}"
echo "[INFO] DATASET_ROOT=${DATASET_ROOT}"
echo "[INFO] CONDA_SH=${CONDA_SH}"
echo "[INFO] CONDA_ENV=${CONDA_ENV}"
echo "[INFO] CKPT_DIR=${CKPT_DIR}"
echo "[INFO] MODEL_SIZE=${MODEL_SIZE} (~10M uses ms=s)"
echo "[INFO] DIRECTION=${DIRECTION}"
echo "[INFO] DATASET=${DATASET}"
echo "[INFO] EPOCHS=${EPOCHS}"
echo "[INFO] NUM_ODE_STEPS=${NUM_ODE_STEPS}"
echo "[INFO] BATCH_SIZE=${BATCH_SIZE}"
echo "[INFO] PATCH_SIZE=${PATCH_H} ${PATCH_W}"
echo "[INFO] ALLOW_FRESH=${ALLOW_FRESH}"
echo "[INFO] Requesting ${NNODES} node, ${GPUS} GPUs, ${CPUS_PER_TASK} CPUs"

trap 'echo; echo "[INFO] Caught interrupt; exiting."; exit 130' INT TERM

ALLOC_IDX=0
TRAINING_FINISHED=0
while (( MAX_ALLOCATIONS == 0 || ALLOC_IDX < MAX_ALLOCATIONS )); do
  ALLOC_IDX=$((ALLOC_IDX + 1))

  if (( MAX_ALLOCATIONS == 0 )); then
    echo "[INFO] Starting allocation window ${ALLOC_IDX}/infinite"
  else
    echo "[INFO] Starting allocation window ${ALLOC_IDX}/${MAX_ALLOCATIONS}"
  fi

  LATEST_CKPT="$(
    find "${CKPT_DIR}" -maxdepth 1 -type f \
      -name "flowformer-${MODEL_SIZE}_${DATASET}_${DIRECTION}_[0-9]*.pth" \
      -print 2>/dev/null \
      | sort -V \
      | tail -n 1
  )"

  echo "[INFO] Looking for checkpoints in: ${CKPT_DIR}"

  if [[ -n "${LATEST_CKPT}" && -s "${LATEST_CKPT}" ]]; then
    echo "[INFO] Found checkpoint: ${LATEST_CKPT}"
    CKPT_ARGS=(--resume --ckpt "${LATEST_CKPT}")
  else
    echo "[WARN] No checkpoint found for ${RUN_NAME}."
    if [[ "${ALLOW_FRESH}" == "1" ]]; then
      echo "[WARN] ALLOW_FRESH=1; starting a new training run."
      CKPT_ARGS=()
    else
      echo "[ERROR] Refusing to start from epoch 1 because ALLOW_FRESH=0."
      echo "[ERROR] For a new run: ALLOW_FRESH=1 bash $0"
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
  --cpus-per-task=$(printf '%q' "${CPUS_PER_TASK}") \
  --kill-on-bad-exit=1 \
  --cpu-bind=cores \
  ${PYTHON_CMD}
INNER_EOF
)

  if salloc "${SALLOC_OPTS[@]}" bash -lc "${ALLOCATION_BODY}"; then
    echo "[INFO] Training finished successfully at $(date)."
    TRAINING_FINISHED=1
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

if (( TRAINING_FINISHED == 1 )); then
  echo "[INFO] Allocation loop ended at $(date)."
  exit 0
fi

echo "[ERROR] Training did not finish successfully."
if (( MAX_ALLOCATIONS > 0 )); then
  echo "[ERROR] Reached MAX_ALLOCATIONS=${MAX_ALLOCATIONS}."
fi
exit 1
