import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from GPT2Modules import SentLevelGPT2Classifier
from DataModules import GPT2DatastoreDataset, GPT2FeatureDataset
from Utils import seed_everything
import torch

def gpt2_get_condition_vector(args):
    def adjust_sentiment_vecs():
        new_sentiment_vectors = np.memmap(args.datastore_condition_vector_path+'new_dstore_'+args.task+'_vectors.npy', dtype=np.float32,
                                          mode='w+', shape=(dstore_size, args.num_labels))
        for index in range(len(dstore_sent_ids)-1):
            if dstore_sent_ids[index] == dstore_sent_ids[index + 1]:
                new_sentiment_vectors[index] = dstore_subseq_condition[index + 1]
            else:
                new_sentiment_vectors[index] = dstore_subseq_condition[index]
    seed_everything(2022)
    if args.dataset == "imdb":
        args.num_labels = 2
    elif args.dataset == "agnews":
        args.num_labels = 4

    model = SentLevelGPT2Classifier(args).to(args.device)
    state_dict = torch.load(args.finetuned_model_save_path+"/best_model.pt")
    model.load_state_dict(state_dict)
    model.eval()

    train_dataset = GPT2DatastoreDataset(args)
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    train_sampler = SequentialSampler(indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    train_loader = tqdm(train_loader, desc="Eval Iteration")

    lens=0
    dstore_size = train_dataset.get_dstore_size()
    dstore_features = np.memmap(args.datastore_condition_vector_path+"dstore_gpt2_"+args.task+"_features.npy", dtype=np.float32, mode='w+', shape=(dstore_size, 1024))
    dstore_sent_ids = np.memmap(args.datastore_condition_vector_path+"dstore_"+args.task+"_sent_ids.npy", dtype=np.int, mode='w+', shape=(dstore_size))
    dstore_subseq_condition = np.memmap(args.datastore_condition_vector_path+"dstore_"+args.task+"_vectors.npy", dtype=np.float32, mode='w+', shape=(dstore_size, args.num_labels))

    for step, batch in enumerate(train_loader):
        inputs = {
            'prev_output_tokens': batch[0],
            'seq_lengths': batch[1],
        }
        with torch.no_grad():
            features, sent_ids = model.extract_feature(**inputs)
            dstore_features[lens:lens+features.shape[0]] = features.to("cpu")#.detach()
            dstore_sent_ids[lens:lens+features.shape[0]] = sent_ids.to("cpu")#.detach()
            lens += features.shape[0]
            #break
    print(args.datastore_path+"一共有"+str(lens)+"条subsequence")

    lens = 0
    feature_dataset = GPT2FeatureDataset(dstore_features)
    dataset_size = len(feature_dataset)
    indices = list(range(dataset_size))
    feature_sampler = SequentialSampler(indices)
    feature_loader = torch.utils.data.DataLoader(feature_dataset, batch_size=1024, sampler=feature_sampler)
    feature_loader = tqdm(feature_loader, desc="Eval Iteration")
    for step, batch in enumerate(feature_loader):
        with torch.no_grad():
            inputs = {
                'word_representation': batch.to(args.device),
            }
            outputs = model.extract_condition_vecs(**inputs).to("cpu")#.detach()
            _, predicted = torch.max(outputs, -1)
            # print("输入 ",batch[0])
            # print("预测打分: ",outputs)
            # print("预测结果: ",predicted)
            dstore_subseq_condition[lens:lens+outputs.shape[0]] = outputs
            lens+=outputs.shape[0]
            #break
    print(lens)
    print("对sentiment vector对应位置进行调整，使vector对应含target词的subseq的sentiment")
    adjust_sentiment_vecs()
    print("结束调整")


