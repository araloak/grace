export CUDA_VISIBLE_DEVICES=0
TASK=single-aspect/agnews/prompts/pplm/
DATASET=gpt2-finetune-topic-prefixs.txt
DATA_BIN=data-bin-gpt2-finetune

#TASK=single-aspect/prompts/sentiment
#DATASET=sentiment_prompts4ctg.txt
#cp ../data/$TASK/$DATASET ../data/$TASK/$DATASET.train
#mv ../data/$TASK/$DATASET ../data/$TASK/$DATASET.val

# BPE text
python ../repo/examples/roberta/multiprocessing_bpe_encoder.py \
            --encoder-json ../data/encoder.json \
            --vocab-bpe ../data/vocab.bpe \
            --inputs ../data/$TASK/$DATASET \
            --outputs ../data/$TASK/$DATASET.bpe \
            --workers 1 \
            --keep-empty;

# Save to .bin file
# gpt2-style data pre-processing
python ../repo/preprocess.py \
    --only-source \
    --trainpref  ../data/$TASK/$DATASET.bpe \
    --validpref ../data/$TASK/$DATASET.bpe \
    --testpref ../data/$TASK/$DATASET.bpe \
    --destdir ../data/$TASK/$DATA_BIN/ \
    --workers 10 \
    --srcdict ../data/dict_gpt2.txt \
