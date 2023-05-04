echo $1
echo $2
echo $3
echo $4
echo $5
echo $6
echo $7
echo $8
echo $9
echo ${10}

# if use freezed gpt2: # knn.freezed.index  dstore_freezed_gpt2
python ../repo/generate.py $5/data-bin-gpt2-pplm \
    --path ../models/gpt2-medium/checkpoint_best.pt \
    --task language_modeling \
    --beam 1 \
    --batch-size 10 \
    --results-path ../data/$1/$2/$6 \
    --gen-subset valid \
    --skip-invalid-size-inputs-valid-test \
    --sampling \
    --nbest 1 \
    --sampling-topk 10 \
    --quiet \
    --sample-break-mode eos \
    --knnlm \
    --indexfile ${10}/$4/knn.freezed.index \
    --dstore-filename ${10}/$4/dstore_freezed_gpt2 \
    --dstore-size $3 \
    --k $8 \
    --gpt2-padding \
    --lmbda $7 \
    --probe 32 \
    --prompt-control-code $5/$6-control-codes.txt \
    --similar-condition-prob $9 \
    --max-len-b 80 \
    --gpt2-gen-mode \
    --classifier-path /g/Projects/CTG/models/gpt2-medium/imdb \
    --pretrained-model-path ../models/gpt2-medium \
    --sentiment-control \
    --min-lmbda ${11} \
    --max-control-step ${12} \
    --refine