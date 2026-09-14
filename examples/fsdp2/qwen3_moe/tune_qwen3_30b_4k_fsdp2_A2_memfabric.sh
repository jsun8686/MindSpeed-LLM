#!/bin/bash
# Qwen3-30B-A3B fine-tuning on A2 with async activation offload to a MemFabric
# remote DRAM pool (DEVICE_RDMA).
#
# Prerequisites (startup order matters, the config store lives on the FAR side):
#   1. On the FIRST remote-memory node, start the FAR daemon hosting the store:
#        python examples/memfabric/memfabric_far_daemon.py \
#            --store-url tcp://10.0.0.1:8572 --with-store \
#            --nic tcp://10.0.0.1:10005 --world-size 16
#   2. On every OTHER remote-memory node, start the remaining FAR daemons:
#        python examples/memfabric/memfabric_far_daemon.py \
#            --store-url tcp://10.0.0.1:8572 \
#            --nic tcp://10.0.1.1:10005 --world-size 16
#   3. Then launch this training script (NEAR side, waits for the store itself).
#
# NOTE: edit mf_store_url / mf_nic / mf_world_size in the yaml to match your
# deployment. Rank ids are auto-assigned by ralloc (auto_ranking), training
# framework rank ids are never used to configure the pool.

source examples/fsdp2/env_config.sh

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6499
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

mkdir -p ./logs
torchrun $DISTRIBUTED_ARGS train_fsdp2.py \
     examples/fsdp2/qwen3_moe/tune_qwen3_30b_4k_fsdp2_A2_memfabric.yaml \
     | tee logs/tune_qwen3_moe_30b_a3b_4K_fsdp2_A2_memfabric_${TIMESTAMP}.log
