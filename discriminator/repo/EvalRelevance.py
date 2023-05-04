from pytorch_transformers import BertTokenizer, BertConfig
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from BertModulesDIYAtten import BertClassifierDIYAtten
from DataModules import EvalRelevanceDataset
from Utils import seed_everything

import torch

#import wandb
#wandb.init(project="eval-condition-relevance")
def eval_relevance(args):

    if args.task == "sentiment" and args.dataset == "imdb":
        args.num_labels = 2
    elif args.task == "topic" and args.dataset == "agnews":
        args.num_labels = 4
    else:
        print("不支持其他condition的检验")
        exit()

    seed_everything(2022)
    bert_tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path, do_lower_case=False)
    config = BertConfig(hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=3072,
                        num_labels=args.num_labels,
                        do_lower_case=False
                       )
    #wandb.config = config
    # Create our custom BERTClassifier model object
    model = BertClassifierDIYAtten(config,args).to(args.device)
    state_dict = torch.load(args.finetuned_model_save_path+"/best_model.pt")
    model.load_state_dict(state_dict)
    model.eval()

    eval_dataset = EvalRelevanceDataset(args,bert_tokenizer)
    dataset_size = len(eval_dataset)
    indices = list(range(dataset_size))
    eval_sampler = SequentialSampler(indices)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, sampler=eval_sampler)
    eval_loader = tqdm(eval_loader, desc="Eval Iteration")
    trueY,testY = [],[]
    with torch.no_grad():
        val_correct_total = 0
        for step, batch in enumerate(eval_loader):
            inputs = {
                'input_ids': batch[0].to(args.device),
                'token_type_ids': batch[1].to(args.device),
                'attention_mask': batch[2].to(args.device),
                'seq_lengths': batch[4].to(args.device),
            }
            labels = batch[3].to(args.device)
            outputs = model(**inputs)

            _, predicted = torch.max(outputs.data, -1)
            # print(predicted)
            # print(labels)
            trueY+=labels.tolist()
            testY+=predicted.tolist()
            correct_reviews_in_batch = (predicted == labels).sum().item()
            val_correct_total += correct_reviews_in_batch
            #wandb.log({"loss": outputs})
            #wandb.watch(model)
        val_acc = val_correct_total * 100 / len(indices)
        from sklearn.metrics import f1_score

        oriF1 = f1_score(trueY, testY, average="macro")
        print("sklearn-f1:", oriF1)
        print("condition relvance: ", val_acc)
        with open(args.val_file_path+".prediction","w",encoding="utf8") as f:
            for a,b in zip(trueY,testY):
                f.write(str(a)+"\t"+str(b)+"\n")