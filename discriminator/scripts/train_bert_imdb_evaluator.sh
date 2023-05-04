TASK=sentiment
DATASET=imdb
BATCH_SIZE=10
NUM_EPOCHS=6
CUDA_ID=0

#python ../repo/main.py \
    --task $TASK \
    --dataset $DATASET \
    --pretrained-model-path ../models/bert-base-cased \
    --finetuned-model-save-path ../models/evaluator/$DATASET \
    --train-file-path ../data/$TASK/$DATASET/imdb4evaluator.txt \
    --datastore-condition-vector-path ../data/$TASK/$DATASET/datastore_sentiment_vectors.pt \
    --datastore-condition-label-path ../data/$TASK/$DATASET/datastore_sentiment_labels.pt \
    --finetuned-model-save-path ../models/evaluator/$DATASET \
    --batch-size $BATCH_SIZE \
    --num-epochs $NUM_EPOCHS \
    --cuda-id $CUDA_ID \
    --do-train

#python ../repo/main.py \
    --task sentiment \
    --dataset imdb \
    --pretrained-model-path ../models/bert-base-cased \
    --finetuned-model-save-path ../models/evaluator/$DATASET \
    --batch-size $BATCH_SIZE \
    --num-epochs $NUM_EPOCHS \
    --cuda-id 0 \
    --do-eval

python ../repo/main.py \
    --task $TASK \
    --dataset $DATASET \
    --pretrained-model-path ../models/bert-base-cased \
    --finetuned-model-save-path ../models/evaluator/$DATASET \
    --train-file-path ../data/$TASK/$DATASET/imdb4ctg.val.csv \
    --val-file-path ../data/$TASK/$DATASET/imdb4ctg.val.csv \
    --datastore-condition-vector-path ../data/$TASK/$DATASET/imdb4ctg_val_sentiment_vectors.pt \
    --datastore-condition-label-path ../data/$TASK/$DATASET/imdb4ctg_val_sentiment_labels.txt \
    --finetuned-model-save-path ../models/evaluator/$DATASET \
    --batch-size $BATCH_SIZE \
    --num-epochs $NUM_EPOCHS \
    --cuda-id $CUDA_ID \
    --do-extract-vector-labels #--debug