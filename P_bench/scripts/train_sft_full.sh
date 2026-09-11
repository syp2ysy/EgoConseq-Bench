#!/usr/bin/env bash
# Qwen3-VL-4B full SFT on four A100s; append native ms-swift overrides.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export ROOT_IMAGE_DIR="$PWD/data/sft/seen_v2"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
SFT_VARIANT="${SFT_VARIANT:-full}"

exec "${SFT_PYTHON:-/home/zhangshan/miniconda3/envs/qwen3vl/bin/python}" -B -m swift.cli.main sft \
  --model "${SFT_MODEL:-Qwen/Qwen3-VL-4B-Instruct}" \
  --use_hf true \
  --dataset "data/sft/seen_v2/json/${SFT_VARIANT}.jsonl" \
  --train_type full \
  --torch_dtype bfloat16 \
  --freeze_llm false \
  --freeze_vit false \
  --freeze_aligner false \
  --learning_rate 1e-5 \
  --vit_lr 1e-6 \
  --aligner_lr 5e-6 \
  --optim adamw_torch \
  --weight_decay 0.01 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --max_grad_norm 1.0 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing true \
  --vit_gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --deepspeed zero3 \
  --attn_impl sdpa \
  --max_length 8192 \
  --lazy_tokenize true \
  --load_from_cache_file true \
  --packing false \
  --split_dataset_ratio 0 \
  --eval_strategy no \
  --dataloader_num_workers 4 \
  --save_steps 1000 \
  --save_total_limit 1 \
  --logging_steps 10 \
  --report_to tensorboard \
  --seed 42 \
  --data_seed 42 \
  --add_version false \
  --output_dir "outputs/sft/qwen3vl-4b-full-finetune/${SFT_VARIANT}" \
  "$@"
