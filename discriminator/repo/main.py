import torch
from pathlib import Path
import os
import argparse

from Classifier import train
from GPT2Classifier import gpt2_train
from DatastorePredictor import get_condition_vector_labels
from GPT2SubseqVecExtractor import gpt2_get_condition_vector
from EvalRelevance import eval_relevance

#os.environ ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--task', default="sentiment", type=str, required=False)
parser.add_argument('--dataset', default="imdb", type=str, required=False)
parser.add_argument('--cuda-id', default=1, type=int, required=False, help='the number of classes for classification tasks')
parser.add_argument('--num-epochs', default=6, type=int, required=False, help='epoch')
parser.add_argument('--gradient-accumulation-steps', default=4, type=int, required=False, help='epoch')
parser.add_argument('--warmup-steps', default=3, type=int, required=False, help='epoch')
parser.add_argument('--lr', default=1e-5, type=float, required=False, help='original static learning rate')
parser.add_argument('--batch-size', default=128, type=int, required=False, help='batch_size')
parser.add_argument('--validation-split', default=0.1, type=float, required=False, help='size of the validation set split from train set before training')
parser.add_argument('--max_seq_len', default=512, type=int, required=False, help='the maximum length of input sentences')
parser.add_argument('--dstore-subseq-size', default=0, type=int, required=False, help='size of dstore')
parser.add_argument('--datastore-condition-vector-path', default="../data/datastore_sentiment_vectors.pt", type=str, required=False, help='path of pre-trained model files')
parser.add_argument('--datastore-condition-label-path', default="../data/datastore_sentiment_labels.txt", type=str, required=False, help='path of pre-trained model files')
parser.add_argument('--datastore-path', default="../../data/datastore.csv", type=str, required=False, help='path of pre-trained model files')
parser.add_argument('--train-file-path', default="", type=str, required=False, help='path of pre-trained model files')
parser.add_argument('--val-file-path', default="", type=str, required=False, help='path of pre-trained model files')
parser.add_argument('--pretrained-model-path', default="../models/bert-base-cased", type=str, required=False, help='path of pre-trained model files')
parser.add_argument('--finetuned-model-save-path', default="../models", type=str, required=False, help='path of pre-trained model files')
parser.add_argument('--content-class', default="None", type=str, required=False, help='path of controlling sentiment for generated text')
parser.add_argument('--shuffle-dataset',action='store_true', required=False, help='choose to pre-train from scratch')
parser.add_argument('--debug', action='store_true', required=False, help='choose to train from pretrained BERT ')
parser.add_argument('--do-train',action='store_true', required=False, help='choose to train from pretrained BERT ')
parser.add_argument('--do-eval', action='store_true', required=False, help='choose to test')
parser.add_argument('--do-extract-vector-labels', action='store_true', required=False, help='choose to test')

args = parser.parse_args()
#os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_id)

args.SEP_TOKEN = '[SEP]'
args.CLS_TOKEN = '[CLS]'
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
print(args.device)
print("Debug",args.debug)

my_file = Path(args.finetuned_model_save_path)
if not my_file.exists():
    os.makedirs(args.finetuned_model_save_path)

topic_vector_path = "../data/datastore_topic_vectors.pt"
topic_label_path = "../data/datastore_topic_labels.txt"

if __name__ == '__main__':
    print(args)
    if args.do_train:
        if "gpt2" in args.pretrained_model_path:
            gpt2_train(args)
        else:
            train(args)
    elif args.do_extract_vector_labels:
        if "gpt2" in args.pretrained_model_path:
            gpt2_get_condition_vector(args)
        else:
            get_condition_vector_labels(args)
    elif args.do_eval:
        eval_relevance(args)



