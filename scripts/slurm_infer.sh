#!/bin/bash
# Supply account/QOS/resources/log paths to sbatch using your site's verified settings.
set -euo pipefail
: "${SLURM_JOB_ID:?Run this script in a Slurm allocation}"
: "${LOGICCON_CONFIG:?Set a configuration file}"
: "${LOGICCON_INPUTS:?Set the public input JSONL}"
: "${LOGICCON_IMAGE_ROOT:?Set the image root directory}"
: "${LOGICCON_OUTPUT:?Set the experiment output root}"
: "${LOGICCON_PYTHON:?Set the installed environment interpreter}"
if [[ -n ${LOGICCON_MODULE:-} ]]; then
    module load "$LOGICCON_MODULE"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
method_shard=${SLURM_ARRAY_TASK_ID:-0}
method_shards=${LOGICCON_NUM_SHARDS:-1}
method_args=(infer --config "$LOGICCON_CONFIG" --inputs "$LOGICCON_INPUTS"
    --image-root "$LOGICCON_IMAGE_ROOT" --output "$LOGICCON_OUTPUT/shard-$method_shard"
    --mode "${LOGICCON_MODE:-full}" --shard-index "$method_shard" --num-shards "$method_shards")
if [[ ${LOGICCON_RESUME:-0} == 1 ]]; then method_args+=(--resume); fi
if [[ -n ${LOGICCON_LIMIT:-} ]]; then method_args+=(--limit "$LOGICCON_LIMIT"); fi
srun "$LOGICCON_PYTHON" -m logiccon.reasoning "${method_args[@]}"
