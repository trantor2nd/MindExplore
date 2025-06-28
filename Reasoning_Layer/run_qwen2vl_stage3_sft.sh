# 27.5GiB * 2
nproc_per_node=4
MASTER_ADDR=127.0.0.1 \
MASTER_PORT=29510 \
CUDA_VISIBLE_DEVICES=0,1,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model /home/data3/zys/MLLM/pretrained_models/Qwen2-VL-7B-Instruct \
    --train_type lora \
    --torch_dtype bfloat16 \
    --dataset /home/data2/zys/Mars_o1/stage_3/cruise_and_sample_train_final.jsonl \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --freeze_aligner false \
    --lora_dropout 0.10 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --save_strategy epoch \
    --logging_steps 5 \
    --gradient_checkpointing false \
    --ddp_find_unused_parameters false \
    --output_dir /home/data2/zys/Mars_o1/stage_3 \
    --attn_impl flash_attn \
    --split_dataset_ratio 0.00 \
    --deepspeed zero2 \
    --data_seed 42 \
    --seed 42 \
    --system 'You are a Mars rover with autonomous consciousness, able to make reasonable and correct action predictions based on images and command data. The predictions involve some detailed data, such as how many steps to move forward and how many steps to walk in a specific direction.'