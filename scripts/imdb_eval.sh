#!/bin/bash
export  CUDA_VISIBLE_DEVICES=0
SEED=1
MODEL_INFIX=datastore/1

#DSTORE_DATA_BIN=data-bin-gpt2-imdb4ctg-val
#DSTORE_SIZE=1450155  #对应data-bin-gpt2-imdb4ctg-val

DSTORE_DATA_BIN=data-bin-freeze-gpt2-ctg-val
DSTORE_SIZE=1410496

# standard lm eval
python ../repo/eval_lm.py /f/$MODEL_INFIX/data-bin-freeze-gpt2-datastore \
    --path ../models/gpt2-medium/checkpoint_best.pt \
    --sample-break-mode eos \
    --batch-size 32 \
    --gen-subset valid \
    --seed $SEED \
    --gpt2-padding \
    --gpt2-gen-mode #source sentence starts without the bos token, we define target sentence as removing the bos token and append an eos token in the back of the source.

# knnlm eval
python ../repo/eval_lm.py ../data/$MODEL_INFIX/data-bin-freeze-gpt2-ctg-val \
    --path ../models/gpt2-medium/checkpoint_best.pt  \
    --gen-subset train \
    --sample-break-mode eos \
    --dstore-filename ../data/$MODEL_INFIX/$DSTORE_DATA_BIN/dstore \
    --indexfile ../data/$MODEL_INFIX/$DSTORE_DATA_BIN/knn.index \
    --batch-size 16 \
    --k 16 \
    --lmbda 0.25 \
    --dstore-size $DSTORE_SIZE \
    --knn-keytype last_ffn_input \
    --probe 32 \
    --knnlm \
    --no-load-keys \
    --knn-sim-func do_not_recomp_l2 \
    --gpt2-padding \
    --gpt2-gen-mode
