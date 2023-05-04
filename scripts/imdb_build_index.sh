#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
SEED=1
MODEL_INFIX=single-aspect/imdb

DATA_BIN=data-bin-freeze-gpt2-ctg-val
DSTORE_SIZE=1317480

#DATA_BIN=data-bin-gpt2-imdb4ctg-val
##DSTORE_SIZE=1450155

WITH_SENT_IDS=1


python ../repo/eval_lm.py ../data/$MODEL_INFIX/$DATA_BIN \
    --path ../models/gpt2-medium/checkpoint_best.pt \
    --gen-subset valid \
    --sample-break-mode eos \
    --batch-size 8 \
    --tokens-per-sample 510 \
    --context-window 510 \
    --dstore-mmap ../data/$MODEL_INFIX/$DATA_BIN/dstore \
    --knn-keytype last_ffn_input \
    --dstore-size $DSTORE_SIZE \
    --save-knnlm-dstore \
    --save-condition-sent-ids \
    --gpt2-padding \
    --gpt2-gen-mode


python ../repo/build_dstore.py \
  --dstore_keys ../data/$MODEL_INFIX/$DATA_BIN/dstore_keys.npy \
  --dstore_vals  ../data/$MODEL_INFIX/$DATA_BIN/dstore_vals.npy \
  --dstore_size $DSTORE_SIZE \
  --faiss_index ../data/$MODEL_INFIX/$DATA_BIN/knn.index \
  --num_keys_to_add_at_a_time 100000 \
  --starting_point 0