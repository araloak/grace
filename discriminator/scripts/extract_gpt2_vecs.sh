export CUDA_VISIBLE_DEVICES=1
TASK=sentiment
DATASET=imdb
BATCH_SIZE=1

python ../repo/main.py \
    --task $TASK \
    --dataset $DATASET \
    --pretrained-model-path ../../KNNLM/models/gpt2-medium \
    --finetuned-model-save-path ../models/gpt2-medium/$DATASET \
    --datastore-condition-vector-path ../../KNNLM/data/datastore_topic/ \
    --datastore-path /f/datastore/topic/datastore.txt.bpe \
    --do-extract-vector-labels \
    --batch-size $BATCH_SIZE

