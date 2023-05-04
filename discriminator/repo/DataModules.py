import re
import torch
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from torch.utils.data import Dataset

#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    text = re.sub(r'[\\]', ' ', text)
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

class SequenceDataset(Dataset):
    def __init__(self, args):
        self.CLS_TOKEN = args.CLS_TOKEN
        self.SEP_TOKEN = args.SEP_TOKEN
        self.max_seq_len = args.max_seq_len
        self.device = args.device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _class, label, text = self.get_label_and_text(index)
        for regex, value_to_replace_with in self.regex_transformations.items():
            text = re.sub(regex, value_to_replace_with, text)

        # Convert input string into tokens with the special BERT Tokenizer which can handle out-of-vocabulary words using subgrams
        # e.g. text = Here is the sentence I want embeddings for.
        #      tokens = [here, is, the, sentence, i, want, em, ##bed, ##ding, ##s, for, .]
        tokens = self.tokenizer.tokenize(text)[:self.max_seq_len - 3]
        seq_length = len(tokens)  # 不考虑CLS与SEP

        # Add [CLS] at the beginning and [SEP] at the end of the tokens list for classification problems
        tokens = [self.CLS_TOKEN] + tokens + [self.SEP_TOKEN]
        # Convert tokens to respective IDs from the vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Segment ID for a single sequence in case of classification is 0.
        segment_ids = [0] * len(input_ids)

        # Input mask where each valid token has mask = 1 and padding has mask = 0
        input_mask = [1] * len(input_ids)

        # padding_length is calculated to reach max_seq_length
        padding_length = self.max_seq_len - len(input_ids)
        input_ids = input_ids + [0] * padding_length
        input_mask = input_mask + [0] * padding_length
        segment_ids = segment_ids + [0] * padding_length

        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len

        # if self.return_text:
        #     return torch.tensor(input_ids, dtype=torch.long, device=self.device), \
        #            torch.tensor(segment_ids, dtype=torch.long, device=self.device), \
        #            torch.tensor(input_mask, device=self.device), \
        #            torch.tensor(label, dtype=torch.long, device=self.device), \
        #            torch.tensor(seq_length, dtype=torch.long, device=self.device), \
        #            text, \
        #            _class
        # else:
        return torch.tensor(input_ids, dtype=torch.long, device=self.device), \
               torch.tensor(segment_ids, dtype=torch.long, device=self.device), \
               torch.tensor(input_mask, device=self.device), \
               torch.tensor(label, dtype=torch.long, device=self.device), \
               torch.tensor(seq_length, dtype=torch.long, device=self.device)

class IMDBSequenceDataset(SequenceDataset):
    def __init__(self, args, tokenizer, regex_transformations={}):
        super().__init__(args)
        # Read JSON file and assign to data variable (list of strings)
        # df = pd.read_json(dataset_file_path, lines=True)
        # df = df.drop(['article_link'], axis=1)
        self.data = pd.read_csv(args.train_file_path)
        self.return_text = False
        # Apply function on review column
        self.data['review'] = self.data['review'].apply(denoise_text)

        self.regex_transformations = regex_transformations
        self.tokenizer = tokenizer
        self.num_class = len(self.data['sentiment'].unique())

    def get_label_and_text(self, index):
        _class = self.data['sentiment'][index]
        label = 1 if _class =='positive' else 0
        text = self.data['review'][index]
        return _class, label, text

class GPT2IMDBSequenceDataset():
    def __init__(self, args, tokenizer, regex_transformations={}):
        self.max_seq_len = 1024
        self.device = args.device
        self.data = pd.read_csv(args.train_file_path)
        self.return_text = False
        # Apply function on review column
        self.data['review'] = self.data['review'].apply(denoise_text)

        self.regex_transformations = regex_transformations
        self.tokenizer = tokenizer
        self.num_class = len(self.data['sentiment'].unique())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _class, label, text = self.get_label_and_text(index)
        for regex, value_to_replace_with in self.regex_transformations.items():
            text = re.sub(regex, value_to_replace_with, text)

        # Convert input string into tokens with the special BERT Tokenizer which can handle out-of-vocabulary words using subgrams
        # e.g. text = Here is the sentence I want embeddings for.
        #      tokens = [here, is, the, sentence, i, want, em, ##bed, ##ding, ##s, for, .]
        input_ids = self.tokenizer.encode(text)[:self.max_seq_len - 1]
        seq_length = len(input_ids)
        # padding_length is calculated to reach max_seq_length
        padding_length = self.max_seq_len - len(input_ids)
        input_ids = input_ids + [50256] * padding_length
        return torch.tensor(input_ids, dtype=torch.long, device=self.device), \
               torch.tensor(label, dtype=torch.long, device=self.device), \
               torch.tensor(seq_length, dtype=torch.long, device=self.device)

    def get_label_and_text(self, index):
        _class = self.data['sentiment'][index]
        label = 1 if _class =='positive' else 0
        text = self.data['review'][index]
        return _class, label, text

class AmazonSequenceDataset(SequenceDataset):
    def __init__(self, args, tokenizer, regex_transformations={}):
        # Read JSON file and assign to data variable (list of strings)
        # df = pd.read_json(dataset_file_path, lines=True)
        # df = df.drop(['article_link'], axis=1)
        self.data = pd.read_csv(args.train_file_path)
        self.return_text = False
        # Apply function on review column
        self.data['Text'] = self.data['Text'].apply(denoise_text)

        self.regex_transformations = regex_transformations
        self.tokenizer = tokenizer
        self.num_class = len(self.data['Score'].unique())

    def get_label_and_text(self, index):
        label = self.data['Score'][index]-1
        if label >2:
            _class = "positive"
        elif label<2:
            _class = "negative"
        else:
            _class = "neural"
        text = self.data['Text'][index]
        return _class, label, text

class AgnewsSequenceDataset(SequenceDataset):
    def __init__(self, args, tokenizer, regex_transformations={}):
        super().__init__(args)
        # Read JSON file and assign to data variable (list of strings)
        # df = pd.read_json(dataset_file_path, lines=True)
        # df = df.drop(['article_link'], axis=1)
        self.data = pd.read_csv(args.train_file_path)
        self.return_text = False
        self.regex_transformations = regex_transformations
        self.tokenizer = tokenizer
        self.num_class = len(self.data['Class Index'].unique())

    def get_label_and_text(self, index):
        label = self.data['Class Index'][index]
        if  label == 1:
            _class = 'world'
        elif  label == 2:
            _class = 'sports'
        elif  label == 3:
            _class = 'business'
        elif  label == 4:
            _class = 'sci/tec'

        train_label = label-1

        text = self.data['Description'][index]
        return _class, train_label, text

class GPT2AgnewsSequenceDataset():
    def __init__(self, args, tokenizer, regex_transformations={}):
        self.max_seq_len = 1024
        self.device = args.device
        self.data = pd.read_csv(args.train_file_path)
        self.return_text = False
        # Apply function on review column
        self.data['Description'] = self.data['Description'].apply(denoise_text)

        self.regex_transformations = regex_transformations
        self.tokenizer = tokenizer
        self.num_class = len(self.data['Class Index'].unique())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _class, label, text = self.get_label_and_text(index)
        for regex, value_to_replace_with in self.regex_transformations.items():
            text = re.sub(regex, value_to_replace_with, text)

        # Convert input string into tokens with the special BERT Tokenizer which can handle out-of-vocabulary words using subgrams
        # e.g. text = Here is the sentence I want embeddings for.
        #      tokens = [here, is, the, sentence, i, want, em, ##bed, ##ding, ##s, for, .]
        input_ids = self.tokenizer.encode(text)[:self.max_seq_len - 1]
        seq_length = len(input_ids)
        # padding_length is calculated to reach max_seq_length
        padding_length = self.max_seq_len - len(input_ids)
        input_ids = input_ids + [50256] * padding_length
        return torch.tensor(input_ids, dtype=torch.long, device=self.device), \
               torch.tensor(label, dtype=torch.long, device=self.device), \
               torch.tensor(seq_length, dtype=torch.long, device=self.device)


    def get_label_and_text(self, index):
        label = self.data['Class Index'][index]
        if  label == 1:
            _class = 'world'
        elif  label == 2:
            _class = 'sports'
        elif  label == 3:
            _class = 'business'
        elif  label == 4:
            _class = 'sci/tec'

        train_label = label-1

        text = self.data['Description'][index]
        return _class, train_label, text

class EvalRelevanceDataset(SequenceDataset):
    def __init__(self,args, tokenizer, regex_transformations={}):
        super().__init__(args)
        # Read JSON file and assign to data variable (list of strings)
        # df = pd.read_json(dataset_file_path, lines=True)
        self.data = [line.strip() for line in open(args.val_file_path,encoding="utf8").readlines()]
        self.data = [x for x in self.data if self.data.count(x) == 1]

        self._class = args.content_class

        try:
            self.data.remove("")
        except:
            pass
        print("dataset size: ",len(self.data))
        self.regex_transformations = regex_transformations
        self.tokenizer = tokenizer
        self.args = args
        # self.topic_keywords = ["science and technology", "business", "sports", "world news", "politics", "economy"]
        # self.sentiment_keywords = ["positive", "negative", "good", "bad" ]

    def get_label_and_text(self, index):
        text = self.data[index]
        text = text.replace("#<|endoftext|>","")
        text = text.replace("<|endoftext|>","")

        if self.args.task == "sentiment":
            if "negative" in self._class:
                label = 0
            elif "positive" in self._class:
                label = 1

        elif self.args.task == "topic":
            if "world" in self._class:
                    label = 0
            elif "sports" in self._class:
                    label = 1
            elif "business" in self._class:
                    label = 2
            elif "tech" in self._class:
                    label = 3

        return self._class, label, text

class GPT2DatastoreDataset():
    def __init__(self, args):
        data = open(args.datastore_path, encoding="utf8").readlines()
        data = [[int(each) for each in line.strip().split()] for line in data]
        print("Dstore number is :\n"+str(len(data)))
        self.max_seq_len = 1024
        self.device = args.device
        self.dstore_size = 0
        self.data = []
        self.batch_size = args.batch_size

        for index, line in enumerate(data):
            if len(line) <= self.max_seq_len:
                self.dstore_size += len(line)
                self.data.append(line)
            else:
                pass
        print("Dstore size of "+args.datastore_path+" is :\n"+str(self.dstore_size))

    def get_label_and_text(self, index):
        text = self.data[index]
        return "None", -1, text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids = self.data[index][:self.max_seq_len]
        seq_length = len(input_ids)
        # padding_length is calculated to reach max_seq_length
        if self.batch_size !=1:
            print("batch size 是1 运行太慢了")
            exit()
            padding_length = self.max_seq_len - len(input_ids)
            input_ids = input_ids + [50256] * padding_length
        else:
            pass
        return torch.tensor(input_ids, dtype=torch.long, device=self.device), \
               torch.tensor(seq_length, dtype=torch.long, device=self.device)

    def get_dstore_size(self):
        return self.dstore_size

class GPT2FeatureDataset():
    def __init__(self, dstore_features):
        self.data = dstore_features

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]
