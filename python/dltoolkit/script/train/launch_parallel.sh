#!/bin/bash

# Distributed training launcher script
# Usage: ./launch_parallel.sh [--nodes=N] [--gpus-per-node=M] [--master-addr=IP] [--master-port=PORT]

# Default parameters
NODES=1
GPUS_PER_NODE=1
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500
TRAIN_SCRIPT="../../tools/train_parallel.py"

# Parse command line arguments
for arg in "$@"
do
    case $arg in
        --nodes=*)
        NODES="${arg#*=}"
        shift
        ;;
        --gpus-per-node=*)
        GPUS_PER_NODE="${arg#*=}"
        shift
        ;;
        --master-addr=*)
        MASTER_ADDR="${arg#*=}"
        shift
        ;;
        --master-port=*)
        MASTER_PORT="${arg#*=}"
        shift
        ;;
        *)
        echo "Unknown argument: $arg"
        exit 1
        ;;
    esac
done

TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

echo "Starting distributed training with:"
echo "- Nodes: $NODES"
echo "- GPUs per node: $GPUS_PER_NODE"
echo "- Total GPUs: $TOTAL_GPUS"
echo "- Master address: $MASTER_ADDR"
echo "- Master port: $MASTER_PORT"

torchrun \
    --nnodes=$NODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $TRAIN_SCRIPT
