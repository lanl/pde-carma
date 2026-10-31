#!/usr/bin/env bash
set -Eeuo pipefail

##### USER TUNABLE DEFAULTS ####################################################
: "${PROJECT_DIR:=/net/sescratch1/exempt/artimis/mrautela/my_projects/pde-dit}"
: "${DATASET_ROOT:=/net/sescratch1/exempt/artimis/mrautela/my_projects/pde-flowformer}"
: "${ENV_PATH:=/net/sescratch1/exempt/artimis/mrautela/pytorch_py310_h100_env}"
: "${CKPT_DIR:=${PROJECT_DIR}/models}"
: "${TRAIN_PY:=train_dit_unidirectional.py}"

# Resources
: "${ACCOUNT:=y26_artimis-test_g}"
: "${NNODES:=1}"
: "${GPUS:=8}"
: "${CPUS_PER_TASK:=48}"
: "${MEM_PER_GPU:=120G}"
: "${WALLTIME:=08:00:00}"
: "${QOS:=large}"

# MAX_ALLOCATIONS=0 means keep requesting new allocations until training finishes.
: "${MAX_ALLOCATIONS:=0}"

# Training config
: "${MODEL_SIZE:=s}"
: "${DIRECTION:=forward}"
: "${DATASET:=ce-rm}"
: "${EPOCHS:=2000}"
: "${RUN_NAME:=dit-adaln_zero-${MODEL_SIZE}_${DATASET}_${DIRECTION}_unidirectional}"

# Safety switch:
# ALLOW_FRESH=0 means do NOT start from epoch 1 if no checkpoint is found.
# Use ALLOW_FRESH=1 only for a truly new run.
: "${ALLOW_FRESH:=0}"

# Current train_dit_unidirectional.py uses DataParallel, so keep this single-node.
if (( NNODES != 1 )); then
  echo "[ERROR] Current train_dit_unidirectional.py uses DataParallel. Set NNODES=1."
  exit 1
fi

PYTHON="${ENV_PATH}/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "[ERROR] python not found at: $PYTHON"
  exit 1
fi

export PATH="${ENV_PATH}/bin:${PATH}"

# Python training flags.
# Resume flags are added later only if a checkpoint is found.
ARGS=(
  --dataset_root "${DATASET_ROOT}"
  --parallel
  --ms "${MODEL_SIZE}"
  --ds "${DATASET}"
  --epochs "${EPOCHS}"
  --dit_arch adaln_zero
)

EXTRA_CONSTRAINT=()

SALLOC_OPTS=(
  -N "${NNODES}"
  --ntasks=1
  --ntasks-per-node=1
  --cpus-per-task="${CPUS_PER_TASK}"
  --gpus-per-node="${GPUS}"
  --hint=nomultithread
  --mem-per-gpu="${MEM_PER_GPU}"
  --time="${WALLTIME}"
  --account="${ACCOUNT}"
  "${EXTRA_CONSTRAINT[@]}"
)

echo "[INFO] Starting salloc chaining loop at $(date)"
echo "[INFO] PROJECT_DIR=${PROJECT_DIR}"
echo "[INFO] DATASET_ROOT=${DATASET_ROOT}"
echo "[INFO] ENV_PATH=${ENV_PATH}"
echo "[INFO] CKPT_DIR=${CKPT_DIR}"
echo "[INFO] MODEL_SIZE=${MODEL_SIZE}"
echo "[INFO] DATASET=${DATASET}"
echo "[INFO] RUN_NAME=${RUN_NAME}"
echo "[INFO] ALLOW_FRESH=${ALLOW_FRESH}"
echo "[INFO] Using python: ${PYTHON}"
echo "[INFO] Requesting ${NNODES} node(s), ${GPUS} GPUs/node, ${CPUS_PER_TASK} CPUs/task"

trap 'echo; echo "[INFO] Caught interrupt; exiting."; exit 130' INT

ALLOC_IDX=0

while (( MAX_ALLOCATIONS == 0 || ALLOC_IDX < MAX_ALLOCATIONS )); do
  ALLOC_IDX=$((ALLOC_IDX + 1))

  if (( MAX_ALLOCATIONS == 0 )); then
    echo "[INFO] Starting allocation window ${ALLOC_IDX}/infinite"
  else
    echo "[INFO] Starting allocation window ${ALLOC_IDX}/${MAX_ALLOCATIONS}"
  fi

  # Pick the newest checkpoint for this run.
  # Pick the checkpoint with the largest epoch number.
  RUN_DIR="${CKPT_DIR}/${RUN_NAME}"

  if [[ -d "${RUN_DIR}" ]]; then
    LATEST_CKPT="$(
      find "${RUN_DIR}" -maxdepth 1 -type f \
        -name "${RUN_NAME}_[0-9]*.pth" \
        -print 2>/dev/null \
        | sort -V \
        | tail -n 1
    )"
  else
    LATEST_CKPT=""
  fi

  echo "[INFO] Looking for checkpoint in:"
  echo "[INFO] ${RUN_DIR}"

  if [[ -n "${LATEST_CKPT}" && -s "${LATEST_CKPT}" ]]; then
    echo "[INFO] Found checkpoint:"
    echo "[INFO] ${LATEST_CKPT}"
    echo "[INFO] Will resume from checkpoint."

    CKPT_ARR=(--resume --ckpt "${LATEST_CKPT}")
  else
    echo "[WARN] No checkpoint found in:"
    echo "[WARN] ${RUN_DIR}"

    if [[ "${ALLOW_FRESH}" == "1" ]]; then
      echo "[WARN] ALLOW_FRESH=1, so starting fresh training."
      CKPT_ARR=()
    else
      echo "[ERROR] Refusing to start fresh because ALLOW_FRESH=0."
      echo "[ERROR] This prevents restarting from epoch 1 and overwriting your run."
      echo "[ERROR] For a truly new run, use:"
      echo "[ERROR] ALLOW_FRESH=1 bash $0"
      exit 1
    fi
  fi

  PYTHON_CMD="$(printf '%q ' "$PYTHON" -u "${PROJECT_DIR}/${TRAIN_PY}" "${ARGS[@]}" "${CKPT_ARR[@]}")"

  echo "[INFO] Python command:"
  echo "${PYTHON_CMD}"

  echo "[INFO] Requesting ${WALLTIME} allocation via salloc..."

  if salloc "${SALLOC_OPTS[@]}" bash -lc "
      set -Eeuo pipefail

      cd '${PROJECT_DIR}'

      echo '[INFO] Running on nodes:'
      scontrol show hostnames \"\$SLURM_JOB_NODELIST\"

      echo \"[INFO] SLURM_JOB_ID=\$SLURM_JOB_ID\"
      echo \"[INFO] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}\"

      export OMP_NUM_THREADS=1

      srun --export=ALL \
        --ntasks=1 \
        --ntasks-per-node=1 \
        --kill-on-bad-exit=1 \
        --cpu-bind=cores \
        bash -lc '$PYTHON_CMD'
    "
  then
    echo "[INFO] Training finished successfully at $(date). Exiting allocation loop."
    break
  else
    rc=$?
    echo "[WARN] Training step ended with code ${rc} at $(date). Sleeping 60s and retrying..."
    sleep 60
  fi
done

if (( MAX_ALLOCATIONS == 0 )); then
  echo "[INFO] Allocation loop ended at $(date)."
else
  echo "[INFO] Reached MAX_ALLOCATIONS=${MAX_ALLOCATIONS}. Exiting at $(date)."
fi
