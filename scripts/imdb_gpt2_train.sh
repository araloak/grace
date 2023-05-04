#!/bin/bash
CUDA_VISIBLE_DEVICES=1
SEED=1

TASK=single-aspect/imdb
DATASET=imdb4ctg.txt

#TASK=single-aspect/agnews
#DATASET=agnews4ctg.txt

python ../repo/train.py G:/Projects/KNNLM/data/$TASK/data-bin-gpt2 \
    --task language_modeling \
    --save-dir ../models/$TASK/gpt2-medium \
    --pre-trained-path ../models/gpt2-medium \
    --arch hf_gpt2_medium \
    --optimizer adam \
    --lr 1e-5 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 1000 \
    --warmup-init-lr 1e-07 \
    --sample-break-mode eos \
    --update-freq 4 \
    --batch-size  1 \
    --criterion cross_entropy \
    --seed $SEED \
    --skip-invalid-size-inputs-valid-test

#    --weight-decay 0.01 \
#    --clip-norm  0.0 \
#    --dropout 0.1 \
#    --tokens-per-sample 512 \
#    --max-update 5000 \

