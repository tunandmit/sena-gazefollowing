# sena-gazefollowing
Sensitive Area (SenA) gaze prediction built on the top Conditional Detection Transformers in retail environments

## Install package
```
pip install -r requirements.txt
```

## Training SenA Gaze model
For single GPU training
```sh
SAVED_PATH='output/finetune'

WANDB_API_KEY='' # Add your wandb show training log

python -m src.train \
    --model_name_or_path 'ponlv/pretrain-vastai-checkpoint-44000' \
    --output_dir $SAVED_PATH \
    --overwrite_output_dir \
    --stage 'finetune' \
    --do_train \
    --dataset_name_train 'ponlv/gaze-following' \
    --do_eval \
    --dataset_name_val 'ponlv/gaze-following-test' \
    --object_path 'src/utils/objects.txt' \
    --num_workers 12 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 3 \
    --learning_rate 2e-5 \
    --weight_decay 1e-2 \
    --warmup_ratio 0.1 \
    --evaluation_strategy 'steps' \
    --save_steps 500 \
    --eval_steps 1000 \
    --save_total_limit 5 \
    --logging_steps 100 \
    --fp16 \
    --remove_unused_columns False \
    --ddp_find_unused_parameters True
```

For multi GPUs on single node, change `CUDA_VISIBLE_DEVICES` and `--num_processes`

```sh
SAVED_PATH='output/finetune'

WANDB_API_KEY='' # Add your wandb show training log

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --main_process_port 11355 \
    --config_file src/accelerate_ds.yml -m src.train \
    --model_name_or_path 'ponlv/pretrain-vastai-checkpoint-44000' \
    --output_dir $SAVED_PATH \
    --overwrite_output_dir \
    --stage 'finetune' \
    --do_train \
    --dataset_name_train 'ponlv/gaze-following' \
    --do_eval \
    --dataset_name_val 'ponlv/gaze-following-test' \
    --object_path 'src/utils/objects.txt' \
    --num_workers 12 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 3 \
    --learning_rate 2e-5 \
    --weight_decay 1e-2 \
    --warmup_ratio 0.1 \
    --evaluation_strategy 'steps' \
    --save_steps 500 \
    --eval_steps 1000 \
    --save_total_limit 5 \
    --logging_steps 100 \
    --fp16 \
    --remove_unused_columns False \
    --ddp_find_unused_parameters True
```

or run with simple setup
```
bash scripts/finetune.sh
```