mkdir -p $1

OMP_NUM_THREADS=8 accelerate launch --main_process_port "$PORT" --config_file ./configs/deepspeed_1gpus_stage0.yaml \
     ./train.py \
    --model_id $MODEL_NAME \
    --quant_model_id $QUANT_MODEL_NAME \
    --gradient_checkpointing ${gradient_checkpointing:-0} \
    --torch_dtype bfloat16 \
    --dataset $DATASET \
    --test_dataset c4,wikitext2 \
    --torch_compile_mode default \
    --remove_unused_columns 0 \
    --do_train \
    --check_stage no_ck \
    --training_type quant_distill \
    --optim adamw_torch_fused \
    --learning_rate $LR \
    --weight_decay 0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --disable_train_zero ${disable_train_zero:-0} \
    --nsamples ${nsamples:-128} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --eval_on_start 0 \
    --eval_strategy epoch \
    --eval_steps 1 \
    --save_strategy epoch \
    --save_only_model 1 \
    --greater_is_better False \
    --logging_strategy steps \
    --logging_steps 1 \
    --include_tokens_per_second \
    --output_dir $1 \
    --num_train_epochs ${num_train_epochs:-1} \
    --seed 0 \
    --run_name $(basename $1) \
    --report_to wandb \
    > $1/training.log 2>&1 