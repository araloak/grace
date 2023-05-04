from pytorch_transformers import BertTokenizer, BertConfig
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from BertModulesDIYAtten import BertClassifierDIYAtten
from DataModules import  EvalRelevanceDataset, IMDBSequenceDataset
from Utils import seed_everything

import torch
def get_condition_vector_labels(args):
    if args.task == "sentiment" and args.dataset == "imdb":
        NUM_LABEL = 2
    elif args.task == "topic" and args.dataset == "agnews":
        NUM_LABEL = 4
    elif args.task == "sentiment" and args.dataset == "amazon":
        NUM_LABEL = 5

    def map_labels(predicteds,labels):
        # label_dict = {"1": "positive", "0": "neural", "-1": "negative", "2": "world", "3": "sports", "4": "business",
        #               "5": "sci/tec", "100": "Not Given"}
        label_dict = {"1": "positive", "0": "negative","-1":"None"}
        predicted_results = []
        if args.task == "sentiment":
            for predicted in predicteds:
                if predicted.item() == 1:
                    predicted_results.append("positive")
                elif predicted.item() == 0:
                    predicted_results.append("negative")
        if args.task == "topic":
            for predicted in predicteds:
                if predicted.item() == 0:
                    predicted_results.append("world")
                elif predicted.item() ==1:
                    predicted_results.append("sports")
                elif predicted.item() == 2:
                    predicted_results.append("business")
                elif predicted.item() == 3:
                    predicted_results.append("sci/tec")
        ground_truth_classes = [label_dict[str(label.item())] for label in labels]
        return predicted_results, ground_truth_classes

    seed_everything(2022)
    bert_tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path, do_lower_case=False)
    config = BertConfig(hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=3072,
                        num_labels=NUM_LABEL,
                        do_lower_case=False
                       )

    # Create our custom BERTClassifier model object
    model = BertClassifierDIYAtten(config,args).to(args.device)
    state_dict = torch.load(args.finetuned_model_save_path+"/best_model.pt")
    model.load_state_dict(state_dict)
    model.eval()

    #train_dataset = EvalRelevanceDataset(args, bert_tokenizer)
    train_dataset = DataStoreSequenceDataset(args, bert_tokenizer)
    #train_dataset = IMDBSequenceDataset(args,bert_tokenizer)
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))

    if args.debug:
        train_sampler = RandomSampler(indices)
    else:
        train_sampler = SequentialSampler(indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)

    if args.debug:
        pass
    else:
        train_loader = tqdm(train_loader, desc="Eval Iteration")
    logits, predicted_labels, ground_truth_classes = [], [], []
    for step, batch in enumerate(train_loader):
        inputs = {
            'input_ids': batch[0].to(args.device),
            'token_type_ids': batch[1].to(args.device),
            'attention_mask': batch[2].to(args.device),
            'seq_lengths': batch[4].to(args.device),
        }
        label = batch[3]
        with torch.no_grad():
            outputs = model(**inputs).to("cpu").detach()
            _, predicted = torch.max(outputs, -1)
        batch_predicted_labels, batch_ground_truth_classes = map_labels(predicted, label)
        if args.debug:
            print("原始文本\n",batch[5])
            print("ground truth: ", batch_ground_truth_classes)
            print("预测打分: ",outputs)
            print("预测结果: ",batch_predicted_labels)
            print("***************************************")
        else:
            logits.append(outputs)
            predicted_labels+=batch_predicted_labels
            ground_truth_classes+=batch_ground_truth_classes
        # if step == 6:
        #     break

    if not args.debug:
        logits = torch.cat(logits, dim = 0)
        print(logits.shape)
        torch.save(logits,args.datastore_condition_vector_path)
        with open(args.datastore_condition_label_path, "w",encoding = "utf8") as f:
            for label in ground_truth_classes:
                f.write(label+"\n")


        #_, predicted = torch.max(logits[0].data, 1)

    # #lines = open(args.input_dir, encoding="utf8").readlines()
    # data = pd.read_csv(datastore_path)
    # text_array = np.array(data["text"])
    # text_list =text_array.tolist()
    #
    # label_array = np.array(data["class"])
    # label_list =label_array.tolist()
    #
    # input_ids = []
    # segment_ids = []
    # input_mask = []
    # seq_lengths = []
    # for words,label in zip(text_list,label_list):
    #     #print(words,label)
    #     tokens = bert_tokenizer.tokenize(words)[:MAX_SEQ_LENGTH-3]
    #     seq_length = len(tokens)
    #     tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]
    #     temp_input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    #
    #     # Segment ID for a single sequence in case of classification is 0.
    #     temp_segment_ids = [0] * len(temp_input_ids)
    #
    #     # Input mask where each valid token has mask = 1 and padding has mask = 0
    #     temp_input_mask = [1] * len(temp_input_ids)
    #
    #     # padding_length is calculated to reach max_seq_length
    #     padding_length = MAX_SEQ_LENGTH - len(temp_input_ids)
    #     temp_input_ids = temp_input_ids + [0] * padding_length
    #     temp_input_mask = temp_input_mask + [0] * padding_length
    #     temp_segment_ids = temp_segment_ids + [0] * padding_length
    #
    #     input_ids += torch.tensor([temp_input_ids], dtype=torch.long, device=DEVICE)
    #     segment_ids += torch.tensor([temp_segment_ids], dtype=torch.long, device=DEVICE)
    #     input_mask += torch.tensor([temp_input_mask], device=DEVICE, dtype=torch.long)
    #     seq_lengths += torch.tensor([seq_length], device=DEVICE, dtype=torch.long)
    #     #print("input_ids", len(input_ids))
    #     if DEBUG and len(input_ids)>10:
    #         break
    #
    # input_ids = torch.stack(input_ids)
    # segment_ids = torch.stack(segment_ids)
    # input_masks = torch.stack(input_mask)
    # seq_lengths = torch.stack(seq_lengths)
    #
    # results = []
    # for input_id, segment_id, input_mask, seq_length in zip(input_ids, segment_ids, input_masks, seq_lengths):
    #     with torch.no_grad():
    #         results.append(
    #             model(input_ids=input_id.unsqueeze(0), token_type_ids=segment_id.unsqueeze(0),
    #                       attention_mask=input_mask.unsqueeze(0), seq_lengths=seq_length.unsqueeze(0)))
    #
    # with open(args.output_dir,"w",encoding="utf8") as f:
    #     for score in results:
    #         f.write(str(score.argmax(-1))+"\n")
    #
    # tso_label_score_list = [each.numpy() for each in tso_label_score_list]
    # tso_label_score_list = np.stack(tso_label_score_list)
    # np.save(tso_label_score_list,args.output+"/"+args.file_name,allow_pickle=True)
