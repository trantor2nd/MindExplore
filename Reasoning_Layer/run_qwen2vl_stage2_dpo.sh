# --rlhf_type cpo/orpo/simpo are also supported
nproc_per_node=4
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29510 \
CUDA_VISIBLE_DEVICES=2,4,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
    --rlhf_type dpo \
    --model /home/data2/zys/Mars_o1/stage_1/v1-20241219-202312/checkpoint-3054-merged \
    --train_type lora \
    --torch_dtype bfloat16 \
    --dataset /home/data2/zys/Mars_o1/stage_2/iter_1_DPO_data_discard_new.jsonl \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-7 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --beta 0.1 \
    --rpo_alpha 1 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --save_strategy epoch \
    --ddp_find_unused_parameters false \
    --output_dir /home/data2/zys/Mars_o1/stage_2/iter_1_DPO \
    --save_total_limit 1 \
    --split_dataset_ratio 0.00 \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --logging_steps 5

# learning_rate 1e-6 5e-7 7e-7
#    --max_length 2048 \