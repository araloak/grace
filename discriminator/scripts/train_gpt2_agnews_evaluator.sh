TASK=topic
DATASET=agnews
BATCH_SIZE=2
NUM_EPOCHS=10
CUDA_ID=0

python ../repo/main.py \
    --task $TASK \
    --dataset $DATASET \
    --pretrained-model-path ../../KNNLM/models/gpt2-medium \
    --finetuned-model-save-path ../models/gpt2-medium/$DATASET \
    --train-file-path ../data/$TASK/$DATASET/agnews4evaluator.csv \
    --finetuned-model-save-path ../models/gpt2-medium/$DATASET \
    --batch-size $BATCH_SIZE \
    --num-epochs $NUM_EPOCHS \
    --cuda-id $CUDA_ID \
    --do-train
