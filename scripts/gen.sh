export CUDA_VISIBLE_DEVICES=1

# configs for topic-controlled generation
#MODEL_INFIX=datastore_topic
#DSTORE_ROOT=/f/datastore/topic
#DSTORE_SIZE=8557170
#REULT_FILE=topic_results/freezed-gpt2/bz_10_bm_1_lmbda_decay_debug
##REULT_FILE=topic_results/freezed-gpt2/world_debug_7
#PROMPT_DATA_DIR=../data/single-aspect/agnews/prompts/pplm

# configs for sentiment-controlled generation
MODEL_INFIX=datastore_sentiment
DSTORE_ROOT=/f/datastore/sentiment
DSTORE_SIZE=11046244
PROMPT_DATA_DIR=../data/single-aspect/imdb/prompts/pplm/repeat

DSTORE_DATA_BIN=data-bin-gpt2-datastore

# demo for topic-controlled generation
#CODE=business
#LMBDA=0.8
#N=0.9
#K=1000
#REULT_FILE=topic_results/finetuned-gpt2
#sh topic_gen.sh $MODEL_INFIX $REULT_FILE $DSTORE_SIZE $DSTORE_DATA_BIN $PROMPT_DATA_DIR $CODE $LMBDA $K $N $DSTORE_ROOT

# demo for sentiment-controlled generation
CODE=positive
MIN_LMBDA=0.4
LMBDA=0.8
N=0.9
K=1000
MAX_CONTROL_STEP=20
REULT_FILE=sentiment_results/freezed-gpt2/ablation/another-ours/new
sh sentiment_gen.sh $MODEL_INFIX $REULT_FILE $DSTORE_SIZE $DSTORE_DATA_BIN $PROMPT_DATA_DIR $CODE $LMBDA $K $N $DSTORE_ROOT $MIN_LMBDA $MAX_CONTROL_STEP
