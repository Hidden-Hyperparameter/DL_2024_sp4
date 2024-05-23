formatted_time=$(date +"%Y%m%d%H%M%S")
echo $formatted_time
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

deepspeed --include localhost:1 finetune.py \
    --model_name_or_path "/ssdshare/LLMs/MiniCPM-SB-dpo-bf16/" \
    --output_dir ./output/AdvertiseGenLoRA/$formatted_time/ \
    --train_data_path ../Datasets/CLS_formatted/train.json \
    --eval_data_path ../Datasets/CLS_formatted/dev.json \
    --learning_rate 5e-5 --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1  --model_max_length 1500 --bf16 --use_lora \
    --gradient_accumulation_steps 4 --warmup_steps 100 \
    --max_steps 4000 --weight_decay 0.01 \
    --evaluation_strategy steps --eval_steps 500 \
    --save_strategy steps --save_steps 500 --seed 42 \
    --log_level info --logging_strategy steps --logging_steps 10 \
    --deepspeed configs/ds_config_zero3_offload.json