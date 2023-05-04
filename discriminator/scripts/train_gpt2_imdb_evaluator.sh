TASK=sentiment
DATASET=imdb
BATCH_SIZE=2
NUM_EPOCHS=6
CUDA_ID=0

python ../repo/main.py \
    --task $TASK \
    --dataset $DATASET \
    --pretrained-model-path ../../KNNLM/models/gpt2-medium \
    --finetuned-model-save-path ../models/gpt2-medium/$DATASET \
    --train-file-path ../data/$TASK/$DATASET/imdb4evaluator.csv \
    --finetuned-model-save-path ../models/gpt2-medium/$DATASET/sent-level \
    --batch-size $BATCH_SIZE \
    --num-epochs $NUM_EPOCHS \
    --cuda-id $CUDA_ID \
    --do-train

#python ../repo/CS/main.py \
    --task sentiment \
    --dataset imdb \
    --pretrained-model-path ../models/bert-base-cased \
    --finetuned-model-save-path ../models/evaluator/$DATASET \
    --batch-size $BATCH_SIZE \
    --num-epochs $NUM_EPOCHS \
    --cuda-id 0 \
    --do-eval
