import numpy as np
import torch.nn as nn
from transformers import GPT2Tokenizer
from pytorch_transformers import WarmupLinearSchedule
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm, trange

from GPT2Modules import SentLevelGPT2Classifier
from DataModules import *
from Utils import seed_everything

import torch
def gpt2_train(args):
    seed_everything()
    # Initialize BERT tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model_path,do_lower_case=False,max_len=args.max_seq_len)
    if args.dataset == "imdb":
        train_dataset = GPT2IMDBSequenceDataset(args, tokenizer)
    elif args.dataset == "agnews":
        train_dataset = GPT2AgnewsSequenceDataset(args, tokenizer)
    args.num_labels = train_dataset.num_class

    model = SentLevelGPT2Classifier(args)
    model.to(args.device)

    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.validation_split * dataset_size))

    if args.shuffle_dataset :
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=validation_sampler)

    print ('Training Set Size {}, Validation Set Size {}'.format(len(train_indices), len(val_indices)))
    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Adam Optimizer with very small learning rate given to BERT
    optimizer = torch.optim.Adam([
        {'params': model.gpt2.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 3e-4},
    ])

    # Learning rate scheduler
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,
                                     t_total=len(train_loader) // args.gradient_accumulation_steps * args.num_epochs)

    model.zero_grad()
    epoch_iterator = trange(int(args.num_epochs), desc="Epoch")
    training_acc_list, validation_acc_list = [], []

    for epoch in epoch_iterator:
        epoch_loss = 0.0
        train_correct_total = 0
        best_val_acc = 0

        # Training Loop
        train_iterator = tqdm(train_loader, desc="Train Iteration")
        for step, batch in enumerate(train_iterator):
            model.train(True)
            # Here each element of batch list refers to one of [input_ids, segment_ids, attention_mask, labels]
            inputs = {
                'prev_output_tokens': batch[0].to(args.device),
                'seq_lengths': batch[2].to(args.device),
            }

            labels = batch[1].to(args.device)
            logits = model(**inputs)

            loss = criterion(logits, labels) / args.gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            _, predicted = torch.max(logits.data, 1)
            correct_reviews_in_batch = (predicted == labels).sum().item()
            train_correct_total += correct_reviews_in_batch
            # if step % 10 == 0:
            #     print(loss)
            #break
        print('Epoch {} - Loss {:.2f}'.format(epoch + 1, epoch_loss / len(train_indices)))

        # Validation Loop
        with torch.no_grad():
            val_correct_total = 0
            model.train(False)
            val_iterator = tqdm(val_loader, desc="Validation Iteration")
            for step, batch in enumerate(val_iterator):
                inputs = {
                    'prev_output_tokens': batch[0].to(args.device),
                    'seq_lengths': batch[2].to(args.device),
                }

                labels = batch[1].to(args.device)
                logits = model(**inputs)

                _, predicted = torch.max(logits.data, 1)
                correct_reviews_in_batch = (predicted == labels).sum().item()
                val_correct_total += correct_reviews_in_batch

                #break
            train_acc = train_correct_total * 100 / len(train_indices)
            val_acc = val_correct_total * 100 / len(val_indices)

            training_acc_list.append(train_acc)
            validation_acc_list.append(val_acc)
            print('Training Accuracy {:.4f} - Validation Accurracy {:.4f}'.format(
                train_acc, val_acc))
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                # 保存
                torch.save(model.state_dict(), args.finetuned_model_save_path+"/gpt2_val_acc_"+str(val_acc)+".pt")
                # 读取
                # the_model.load_state_dict(torch.load(PATH))

