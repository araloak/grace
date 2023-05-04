export CUDA_VISIBLE_DEVICES=1
MODEL_INFIX=database_3
REULT_FILE=new_results

DSTORE_SIZE=17418457
DSTORE_DATA_BIN=data-bin-freeze-gpt2-datastore

PROMPT_DATA_DIR=../data/single-aspect/imdb/prompts/pplm
FINETUNE_MODEL_DIR=../models/gpt2-medium

LMBDA=0.8
K=1024
N=512
#运行前检查一下generate-valid-$GEN_NUM.txt是否已经存在！！！
# no knnlm gen
#python ../repo/generate.py $PROMPT_DATA_DIR/data-bin-gpt2-pplm \
  --task language_modeling \
  --path $FINETUNE_MODEL_DIR/checkpoint_best.pt \
  --batch-size 4 \
  --results-path $FINETUNE_MODEL_DIR/results/negative \
  --skip-invalid-size-inputs-valid-test \
  --quiet \
  --sampling \
  --nbest 3 \
  --beam 3 \
  --sampling-topk 10 \
  --sample-break-mode eos \
  --gpt2-padding \
  --gen-subset valid \
  --max-len-b 80 \
  --gpt2-gen-mode

# knnlm gen
#python ../repo/generate.py $PROMPT_DATA_DIR/data-bin-gpt2-pplm \
    --path $FINETUNE_MODEL_DIR/checkpoint_best.pt \
    --task language_modeling \
    --beam 3 \
    --batch-size 4 \
    --results-path $FINETUNE_MODEL_DIR/$MODEL_INFIX/results/negative \
    --gen-subset valid \
    --skip-invalid-size-inputs-valid-test \
    --sampling \
    --nbest 3 \
    --sampling-topk 10 \
    --quiet \
    --sample-break-mode eos \
    --knnlm \
    --indexfile ../data/$MODEL_INFIX/$DSTORE_DATA_BIN/knn.index \
    --dstore-filename ../data/$MODEL_INFIX/$DSTORE_DATA_BIN/dstore \
    --dstore-size $DSTORE_SIZE \
    --k $K \
    --lmbda $LMBDA \
    --gpt2-padding \
    --max-len-b 80 \
    --gpt2-gen-mode
