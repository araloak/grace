#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
SEED=1

CODE=positive
#CONTROL_CODE=sentiment_results/freezed-gpt2/ablation/another-ours/$CODE
CONTROL_CODE=datastore_sentiment/sentiment_results/freezed-gpt2/ablation/another-ours/new/positive
#CONTROL_CODE=aa/sentiment_control/
OUTPUT=generate-valid-knnlm-k-1000-lmbda-0.8-sentiment-control-0.9.txt


TASK=imdb
echo $CONTROL_CODE
echo $OUTPUT
python ../../discriminator/repo/main.py \
    --task sentiment \
    --dataset imdb \
    --val-file-path ../data/$CONTROL_CODE/$OUTPUT \
    --pretrained-model-path ../../CTG/models/bert-base-cased \
    --finetuned-model-save-path ../../CTG/models/evaluator/$TASK \
    --batch-size 8 \
    --cuda-id 0 \
    --do-eval \
    --content-class $CODE