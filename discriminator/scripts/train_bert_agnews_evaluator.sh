TASK=topic
DATASET=agnews
BATCH_SIZE=6
NUM_EPOCHS=6
CUDA_ID=1

python ../repo/main.py \
    --task $TASK \
    --dataset $DATASET \
    --pretrained-model-path ../models/bert-base-cased \
    --finetuned-model-save-path ../models/evaluator/$DATASET \
    --train-file-path ../data/$TASK/$DATASET/agnews4evaluator.csv \
    --finetuned-model-save-path ../models/evaluator/$DATASET \
    --batch-size $BATCH_SIZE \
    --num-epochs $NUM_EPOCHS \
    --cuda-id $CUDA_ID \
    --do-train
