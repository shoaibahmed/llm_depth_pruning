#!/bin/bash

export TOKENIZERS_PARALLELISM=false  # disable tokenizer warning
pip install wget

# Get the DDP args
HEAD_NODE_IP=$1
NUM_NODES=$2
NUM_GPUS_PER_NODE=2
echo "Head node IP: ${HEAD_NODE_IP} / # nodes: ${NUM_NODES} / # GPUs per node: ${NUM_GPUS_PER_NODE}"

# Check if HEAD_NODE_IP is given
if [ -z "${HEAD_NODE_IP}" ]; then
    echo "No head node IP found. Using torchrun runner."
    RUNNER_CMD="torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS_PER_NODE}"
else
    export WORLD_SIZE=${SLURM_NTASKS}
    export RANK=${SLURM_PROCID}
    export LOCAL_RANK=${SLURM_LOCALID}
    export MASTER_ADDR=${HEAD_NODE_IP}
    export MASTER_PORT=29500
    echo "python args / world size: ${WORLD_SIZE} / rank: ${RANK} / local rank: ${LOCAL_RANK} / master addr: ${MASTER_ADDR} / master port: ${MASTER_PORT}"

    RUNNER_CMD="python"
fi

DEFAULT_MODEL="llama-2"
MODEL=${3:-$DEFAULT_MODEL}
echo "Using model: ${MODEL}"

DEFAULT_PRUNING_SCHEME="mhsa"
PRUNING_SCHEME=${4:-$DEFAULT_PRUNING_SCHEME}
echo "Pruning scheme: ${PRUNING_SCHEME}"

${RUNNER_CMD} layer_influence/evaluate_layer_influence.py \
    --dataset "openwebtext" \
    --model-name ${MODEL} \
    --model-size 7b \
    --batch-size 1 \
    --sequence-length 2048 \
    --subsample-size 250000 \
    --pruning-scheme ${PRUNING_SCHEME} \
    --limit-shapley-samples 25000 \
    --wandb-project 'layer_influence'
