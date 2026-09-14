#!/bin/bash
# qwen3-30b-a3b SFT with generic async activation offload to a MemFabric remote
# DRAM pool (mcore backend, TP/DP+EP, no PP/VPP/CP per v1 scope).
#
# Deployment order (auto-rank mode, store hosted on the FAR side):
#   1. start the first FAR daemon (hosts the config store):
#        python examples/memfabric/memfabric_far_daemon.py \
#            --store-url tcp://<far_ip>:8572 --with-store \
#            --nic tcp://<far_ip>:10005 --world-size 16
#   2. start the remaining FAR daemons on the other memory nodes
#   3. launch this training script (NEAR side)
#
# Prerequisites: pip install memfabric_hybrid; RoCE NIC reachable; the FAR-side
# nodes contribute DRAM (one FAR daemon process per memory node).
# NOTE: --recompute-granularity full is rejected with --activation-offload in v1
# (hook nesting with mcore checkpointing unverified); leave recompute off or use
# selective.

export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True


NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="your model ckpt path"
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"

# MemFabric remote pool configuration (FAR-side addresses)
MF_STORE_URL="tcp://10.0.0.1:8572"
MF_NIC="tcp://10.0.0.1:10005"
MF_WORLD_SIZE=16

TP=8
PP=1
EP=8
SEQ_LENGTH=4096
TRAIN_ITERS=2000

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 768 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-router-load-balancing-type aux_loss \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
    --moe-aux-loss-coeff 0.001
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --sequence-parallel \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer
"

ACTIVATION_OFFLOAD_ARGS="
    --activation-offload \
    --activation-offload-backend memfabric \
    --offload-pool-size-gb 8 \
    --offload-extend-block-gb 1 \
    --offload-register-mode per_tensor \
    --mf-store-url ${MF_STORE_URL} \
    --mf-nic ${MF_NIC} \
    --mf-world-size ${MF_WORLD_SIZE} \
    --mf-pool-id 0 \
    --mf-store-wait-timeout 300
"

TRAIN_ARGS="
    --micro-batch-size 4 \
    --global-batch-size 32 \
    --lr 1.25e-5 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --no-shared-storage
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP}
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --norm-topk-prob \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 48 \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 4 \
    --ckpt-format torch
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng
"

TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --tokenizer-not-use-fast \
    --prompt-type qwen3 \
    --no-pad-to-seq-lengths
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $TUNE_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $ACTIVATION_OFFLOAD_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --load ${CKPT_LOAD_DIR} \
	--save ${CKPT_SAVE_DIR} \
    --distributed-backend nccl \
    --transformer-impl local \
    | tee logs/tune_qwen3_30b_a3b_memfabric.log
