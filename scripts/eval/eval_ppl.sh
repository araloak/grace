#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
SEED=1
MODEL_INFIX=../data/main_results/sentiment
MODEL_INFIX=../data/datastore_sentiment/sentiment_results/freezed-gpt2/lmbda_0.8-0.4-retrieve_in_20
MODEL_INFIX=../data/aa_final/variants
MODEL_INFIX=../data/PCTG
MODEL_INFIX=../data/COCON


CONTROL_CODE=computer
EVAL_DATA_BIN=data-bin-gpt2-eval-ppl
DATASET=help_the_development_of_economy.txt
echo $MODEL_INFIX
echo $CONTROL_CODE
echo $DATASET
# 去除空的换行符
python ../repo/post_process.py \
            --input $MODEL_INFIX/$CONTROL_CODE/$DATASET \
            --output $MODEL_INFIX/$CONTROL_CODE/$DATASET.post

python ../repo/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../data/encoder.json \
            --vocab-bpe ../data/vocab.bpe \
            --inputs $MODEL_INFIX/$CONTROL_CODE/$DATASET.post \
            --outputs $MODEL_INFIX/$CONTROL_CODE/$DATASET.post.bpe \
            --workers 1 \
            --keep-empty;

# Save to .bin file
# set gpt2_style as True to perform tokenization in gpt style
python ../repo/preprocess.py \
    --only-source \
    --trainpref  $MODEL_INFIX/$CONTROL_CODE/$DATASET.post.bpe \
    --validpref $MODEL_INFIX/$CONTROL_CODE/$DATASET.post.bpe \
    --testpref $MODEL_INFIX/$CONTROL_CODE/$DATASET.post.bpe \
    --destdir $MODEL_INFIX/$CONTROL_CODE/$EVAL_DATA_BIN/ \
    --workers 1 \
    --srcdict ../data/dict_gpt2.txt \

# lm eval
python ../repo/eval_lm.py $MODEL_INFIX/$CONTROL_CODE/$EVAL_DATA_BIN \
    --path ../models/gpt2-large/checkpoint_best.pt \
    --sample-break-mode eos \
    --batch-size 4 \
    --gen-subset valid \
    --seed $SEED \
    --gpt2-padding \
    --gpt2-gen-mode
