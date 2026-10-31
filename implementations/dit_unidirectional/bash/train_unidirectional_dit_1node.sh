#!/usr/bin/env bash
set -Eeuo pipefail

##### USER TUNABLE DEFAULTS ####################################################
: "${PROJECT_DIR:=/users/rupasree/pde-dit}"
: "${DATASET_ROOT:=/net/sescratch1/exempt/artimis/rupasree}"
: "${ENV_PATH:=/users/rupasree/miniconda3/envs/pytorch_py312_env}"
: "${CKPT_DIR:=${PROJECT_DIR}/models}"
: "${TRAIN_PY:=train_dit_unidirectional.py}"

# Resources
: "${NNODES:=1}"
: "${GPUS:=8}"
: "${CPUS_PER_TASK:=48}"
: "${MEM_PER_GPU:=120G}"
: "${WALLTIME:=08:00:00}"
: "${PARTITION:=standard}"
: "${QOS:=standard}"

# 0 means continue requesting allocations until training finishes.
: "${MAX_ALLOCATIONS:=0}"

# Training configuration
: "${MODEL_SIZE:=m}"
: "${DATASET:=ce-rm}"
: "${EPOCHS:=2000}"
: "${BATCH_SIZE:=256}"
: "${SAMP_DIFF_STEPS:=100}"
: "${RUN_NAME:=dit-${MODEL_SIZE}_${DATASET}_forward_unidirectional}"

# Set to 1 only for a new run.
: "${ALLOW_FRESH:=0}"

if (( NNODES != 1 )); then
    echo "[ERROR] DataParallel requires NNODES=1."
    exit 1
fi

PYTHON="${ENV_PATH}/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
    echo "[ERROR] Python not found at: ${PYTHON}"
    exit 1
fi

if [[ ! -f "${PROJECT_DIR}/${TRAIN_PY}" ]]; then
    echo "[ERROR] Training script not found:"
    echo "[ERROR] ${PROJECT_DIR}/${TRAIN_PY}"
    exit 1
fi

export PATH="${ENV_PATH}/bin:${PATH}"

ARGS=(
    --dataset_root "${DATASET_ROOT}"
    --parallel
    --ms "${MODEL_SIZE}"
    --ds "${DATASET}"
    --epochs "${EPOCHS}"
    --bs "${BATCH_SIZE}"
    --samp_diff_steps "${SAMP_DIFF_STEPS}"
)

SALLOC_OPTS=(
    --nodes="${NNODES}"
    --ntasks=1
    --ntasks-per-node=1
    --cpus-per-task="${CPUS_PER_TASK}"
    --gpus-per-node="${GPUS}"
    --partition="${PARTITION}"
    --qos="${QOS}"
    --hint=nomultithread
    --mem-per-gpu="${MEM_PER_GPU}"
    --time="${WALLTIME}"
)

echo "[INFO] Starting allocation loop at $(date)"
echo "[INFO] PROJECT_DIR=${PROJECT_DIR}"
echo "[INFO] DATASET_ROOT=${DATASET_ROOT}"
echo "[INFO] ENV_PATH=${ENV_PATH}"
echo "[INFO] TRAIN_PY=${TRAIN_PY}"
echo "[INFO] RUN_NAME=${RUN_NAME}"
echo "[INFO] PARTITION=${PARTITION}"
echo "[INFO] QOS=${QOS}"
echo "[INFO] ALLOW_FRESH=${ALLOW_FRESH}"

trap 'echo; echo "[INFO] Caught interrupt; exiting."; exit 130' INT

ALLOC_IDX=0

while (( MAX_ALLOCATIONS == 0 || ALLOC_IDX < MAX_ALLOCATIONS )); do
    ALLOC_IDX=$((ALLOC_IDX + 1))

    if (( MAX_ALLOCATIONS == 0 )); then
        echo "[INFO] Starting allocation ${ALLOC_IDX}/infinite"
    else
        echo "[INFO] Starting allocation ${ALLOC_IDX}/${MAX_ALLOCATIONS}"
    fi

    RUN_DIR="${CKPT_DIR}/${RUN_NAME}"

    # Expected checkpoint examples:
    # dit-m_ce-rm_forward_unidirectional_1.pth
    # dit-m_ce-rm_forward_unidirectional_500.pth
    LATEST_CKPT="$(
        ls -t \
            "${RUN_DIR}"/dit-"${MODEL_SIZE}"_"${DATASET}"_forward_unidirectional_*.pth \
            2>/dev/null |
            head -n 1 || true
    )"

    echo "[INFO] Looking for checkpoint in:"
    echo "[INFO] ${RUN_DIR}"

    if [[ -n "${LATEST_CKPT}" && -s "${LATEST_CKPT}" ]]; then
        echo "[INFO] Found checkpoint: ${LATEST_CKPT}"
        CKPT_ARGS=(--resume --ckpt "${LATEST_CKPT}")
    else
        echo "[WARN] No checkpoint found in ${RUN_DIR}"

        if [[ "${ALLOW_FRESH}" == "1" ]]; then
            echo "[WARN] Starting a fresh run."
            CKPT_ARGS=()
        else
            echo "[ERROR] Refusing to start fresh because ALLOW_FRESH=0."
            echo "[ERROR] For a new run, use ALLOW_FRESH=1."
            exit 1
        fi
    fi

    printf -v PYTHON_CMD '%q ' \
        "${PYTHON}" \
        -u \
        "${PROJECT_DIR}/${TRAIN_PY}" \
        "${ARGS[@]}" \
        "${CKPT_ARGS[@]}"

    echo "[INFO] Python command:"
    echo "${PYTHON_CMD}"

    if salloc "${SALLOC_OPTS[@]}" bash -lc "
        set -Eeuo pipefail

        cd '${PROJECT_DIR}'

        echo '[INFO] Running on nodes:'
        scontrol show hostnames \"\$SLURM_JOB_NODELIST\"

        echo \"[INFO] SLURM_JOB_ID=\$SLURM_JOB_ID\"
        echo \"[INFO] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}\"

        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1

        srun \
            --export=ALL \
            --ntasks=1 \
            --ntasks-per-node=1 \
            --kill-on-bad-exit=1 \
            --cpu-bind=cores \
            bash -lc '${PYTHON_CMD}'
    "
    then
        echo "[INFO] Training completed successfully at $(date)."
        break
    else
        rc=$?
        echo "[WARN] Training ended with code ${rc} at $(date)."
        echo "[WARN] Sleeping for 60 seconds before retrying."
        sleep 60
    fi
done