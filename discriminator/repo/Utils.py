import os
import random

import numpy as np
import torch


def seed_everything(seed = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def gpt2_encode_sents():
    text_bpe_path = r"F:\datastore\imdb4ctg\imdb4ctg.txt.val.bpe"
    dstore_path = r"F:\datastore\imdb4ctg\imdb4ctg_subseq_ids.npy"

    text_bpe = open(text_bpe_path, encoding="utf8").readlines()
    text_bpe = [[int(each) for each in line.strip().split()] for line in text_bpe]
    print("total sentence number: ",len(text_bpe))
    vecs = []
    dstore_size = 0

    for index, line in enumerate(text_bpe):
        if len(line) <= 1024:
            dstore_size += len(line)
        else:
            pass
    dstore_ids = np.memmap(dstore_path, dtype=np.int32, mode='w+', shape=(dstore_size, 1024))

    lens = 0
    for index, line in enumerate(text_bpe):
        if len(line) <= 1024:
            for i in range(min(len(line), 1024)):
                input_ids = np.full((1, 1024), 50256)
                input_ids[0][0:i + 1] = np.array(line[0:i + 1])
                dstore_ids[lens] = input_ids[0]
                lens += 1
        else:
            pass

        if index % 10000 == 0:
            print(index, "/", len(text_bpe))
    print(lens)

