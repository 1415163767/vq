#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero2.json

# Model configuration
llm=Qwen/Qwen3-VL-2B-Instruct  # Using HuggingFace model ID
# llm=/blob/dyb_output/important/qwen3_vq_initialize_checkpoint-1

# Training hyperparameters
lr=1e-4
batch_size=1
grad_accum_steps=1

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Output configuration
run_name="multiple_codebook_ema_scale_image_video"
output_dir=/blob/dyb_output/icml2026/multiple_codebook_ema_scale_image_video
export WANDB_PROJECT="icml_2026_vq_ablation"

# Training arguments
args="
    --model_name_or_path "${llm}" \
    --train_vq_wo_llm True \
    --add_image_data True \
    --add_video_data True \
    --show_data_structure True \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_vqvae True \
    --tune_mm_llm False \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 8388608 \
    --min_pixels 262144 \
    --video_max_pixels 33554432 \
    --video_min_pixels 1048576 \
    --video_max_frames 32 \
    --video_min_frames 4 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 100 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nnodes=4 \
         --nproc_per_node=8 \
         --node_rank=2 \
         --master_addr=100.65.133.85 \
         --master_port=29500 \
         ${entry_file} ${args}

python /blob/thinking.py
