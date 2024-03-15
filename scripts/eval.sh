SAVED_PATH='output/result'

rm -rf $SAVED_PATH

CUDA_VISIBLE_DEVICES=0 python -m src.evaluate \
    --model_name_or_path 'weights/finetune' \
    --dataset_name_test 'data/gaze-real' \
    --object_path 'src/utils/objects.txt' \
    --stage 'finetune'
