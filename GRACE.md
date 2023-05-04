# GRACE

This code is for the Findings of ACL 2023 paper: GRACE: Gradient-guided Controllable Retrieval for Augmenting Attribute-based Text Generation. If you use this code or results from our paper, please cite:

```
@inproceedings{GRACE,
    title = "GRACE: Gradient-guided Controllable Retrieval for Augmenting Attribute-based Text Generation",
    author = "Zhihua Wen, Zhiliang Tian, Zhen Huang, Yuxin Yang, Zexin Jian, Changjian Wang and Dongsheng Li",
    booktitle = "Findings of ACL 2023",
}
```

## Setup

### Generator

1. Please download [GPT2-medium](https://huggingface.co/GPT2-medium) for generation, which can be replaced by any type of auto-regressive generation models (e.g., GPT2-xl and gpt3).
2. Convert the downloaded `pytorch_model.bin` into `checkpoint_best.pt` file required by Fairseq and save the checkpoint in `models/gpt2-medium`.

### Discriminator

1. We split the attribute classification datasets ([IMDB](https://ai.stanford.edu/~amaas/data/sentiment/) and [AG news](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) in our work) and use only the half of each dataset to train the discriminator and build the  retrieval repository. (another half is for training evaluator)
2.  [`train_GPT2_imdb_evaluator.sh`](discriminator/scripts/train_GPT2_imdb_evaluator.sh) and [`train_GPT2_agnews_evaluator.sh`]() fine-tune GPT2-medium to obtain sentiment and topic discriminator respectively. They are used to extract attribute-augmented context representations for unlabelled corpus.

We will release our fine-tuned checkpoints soon.

### Retrieval Repository

1. Please follow [urvashik/knnlm](https://github.com/urvashik/knnlm) to build the retrieval repository. We make minor modifications in [`build_dstore.py`](repo/build_dstore.py) (we support sub-sentence level representation extraction instead of a fix-length window over the whole document).
2. Use [`extract_gpt2_vecs.sh`](discriminator/scripts/extract_gpt2_vecs.sh) to extract attribute-augmented context representations.

### Evaluation

1. [`train_imdb_evaluator.sh`](discriminator/scripts/train_imdb_evaluator.sh) and [`train_agnews_evaluator.sh`](discriminator/scripts/train_agnews_evaluator.sh) use another half of the attribute datasets to train BERT-based evaluators to evaluate the attribute expressed by GRACE.
2. We also use existing Huggingface [sentiment classifier](https://huggingface.co/gchhablani/bert-base-cased-finetuned-sst2) to evaluate sentiment accuracy.

## Usage

1. Run [`sentimen_gen.sh`](scripts/sentiment_gen.sh) and [`topic_gen.sh`](scripts/topic_gen.sh) to generate sentences for different attributes, where `--refine` is to allow gradient-based generation, `–-k` controls the number of retrieval results, `--similar-condition-prob` echoes the threshold $p$ in our paper, and `--max-control-step` defines the maximum number of retrieval steps.
2. For sentiment-controlled generation, we support `positive` and `negative` sentiment. For topic-controlled generation, we support `business`, `polities`, `technology`, and `world news (world)`.
