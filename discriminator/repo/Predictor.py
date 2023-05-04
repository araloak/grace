import numpy as np
import pandas as pd
import torch.nn as nn
from pytorch_transformers import BertTokenizer, BertConfig
import torch

from BertModulesDIYAtten import BertClassifierDIYAtten
from DataModules import SequenceDataset, denoise_text
from Utils import seed_everything

def test(args):
    if args.task == "sentiment" and args.dataset == "imdb":
        NUM_LABEL = 2
    elif args.task == "topic" and args.dataset == "agnews":
        NUM_LABEL = 4
    elif args.task == "sentiment" and args.dataset == "amazon":
        NUM_LABEL = 5

    seed_everything()
    bert_tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path, do_lower_case=False)
    config = BertConfig(hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=3072,
                        num_labels=NUM_LABEL,
                        do_lower_case=False
                       )

    # Create our custom BERTClassifier model object
    model = BertClassifierDIYAtten(config,args.pretrained_model_path).to(args.device)
    state_dict = torch.load(args.finetuned_model_save_path+"/best_model.pt")
    model.load_state_dict(state_dict)
    model.eval()

    #lines = open(args.input_dir, encoding="utf8").readlines()
    data = pd.read_csv(args.datastore_path)
    text_array = np.array(data["text"])
    text_list =text_array.tolist()

    label_array = np.array(data["class"])
    label_list =label_array.tolist()

    input_ids = []
    segment_ids = []
    input_mask = []
    seq_lengths = []
    for words,label in zip(text_list,label_list):
        #print(words,label)
        tokens = bert_tokenizer.tokenize(words)[:args.max_seq_len-3]
        seq_length = len(tokens)
        tokens = [args.CLS_TOKEN] + tokens + [args.SEP_TOKEN]
        temp_input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)

        # Segment ID for a single sequence in case of classification is 0.
        temp_segment_ids = [0] * len(temp_input_ids)

        # Input mask where each valid token has mask = 1 and padding has mask = 0
        temp_input_mask = [1] * len(temp_input_ids)

        # padding_length is calculated to reach max_seq_length
        padding_length = args.max_seq_len - len(temp_input_ids)
        temp_input_ids = temp_input_ids + [0] * padding_length
        temp_input_mask = temp_input_mask + [0] * padding_length
        temp_segment_ids = temp_segment_ids + [0] * padding_length

        input_ids += torch.tensor([temp_input_ids], dtype=torch.long, device=args.device)
        segment_ids += torch.tensor([temp_segment_ids], dtype=torch.long, device=args.device)
        input_mask += torch.tensor([temp_input_mask], device=args.device, dtype=torch.long)
        seq_lengths += torch.tensor([seq_length], device=args.device, dtype=torch.long)
        #print("input_ids", len(input_ids))
        if args.DEBUG and len(input_ids)>10:
            break

    input_ids = torch.stack(input_ids)
    segment_ids = torch.stack(segment_ids)
    input_masks = torch.stack(input_mask)
    seq_lengths = torch.stack(seq_lengths)

    results = []
    for input_id, segment_id, input_mask, seq_length in zip(input_ids, segment_ids, input_masks, seq_lengths):
        with torch.no_grad():
            results.append(
                model(input_ids=input_id.unsqueeze(0), token_type_ids=segment_id.unsqueeze(0),
                          attention_mask=input_mask.unsqueeze(0), seq_lengths=seq_length.unsqueeze(0)))
    #
    # with open(args.output_dir,"w",encoding="utf8") as f:
    #     for score in results:
    #         f.write(str(score.argmax(-1))+"\n")
    #
    # tso_label_score_list = [each.numpy() for each in tso_label_score_list]
    # tso_label_score_list = np.stack(tso_label_score_list)
    # np.save(tso_label_score_list,args.output+"/"+args.file_name,allow_pickle=True)
