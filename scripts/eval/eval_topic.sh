#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
SEED=1

CODE=business
#CONTROL_CODE=topic_results/freezed-gpt2/ablation/another-ours/$CODE
#CONTROL_CODE=main_results/topic/finetuned-gpt2/gpt2
CONTROL_CODE=mm/topic/
OUTPUT=business-20.txt

TASK=agnews
echo $CONTROL_CODE
echo $OUTPUT
python ../../../discriminator/repo/main.py \
    --task topic \
    --dataset agnews \
    --val-file-path ../data/$CONTROL_CODE/$OUTPUT \
    --pretrained-model-path ../../CTG/models/bert-base-cased \
    --finetuned-model-save-path ../../CTG/models/evaluator/$TASK \
    --batch-size 8 \
    --cuda-id 0 \
    --do-eval \
    --content-class $CODE



