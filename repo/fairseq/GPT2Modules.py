import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model
import numpy as np

class TokenLevelGPT2Classifier(nn.Module):
    def __init__(self, args):
        super(TokenLevelGPT2Classifier, self).__init__()
        config = GPT2Config.from_pretrained(args.pretrained_model_path)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(args.pretrained_model_path)
        self.device = args.device
        self.num_labels = config.num_labels

        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(0.1)

        self.V = nn.Parameter(torch.randn(config.hidden_size, 1))

        self.softmax = nn.Softmax(dim=-1)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # set zero embedding for padding symbol
        self.return_all = True
        self.pad_idx = 50256
        self.eos_idx = 2
        self.bos_idx = 2
        #self.pad_idx = task.target_dictionary.pad()
        #self.model.transformer.wte.weight.data[self.pad_idx].zero_()
        #self.model.transformer.wpe.weight.data[0].zero_()

    def forward(
        self,
        prev_output_tokens,
        src_lengths=None,
        incremental_state = None,
        encoder_out=None,
        args=None,
        seq_lengths= None
    ):
        if incremental_state:
            past_key_values = self.get_incremental_state("past_key_values")
        else:
            past_key_values = None

        # don't attend to padding symbols
        attention_mask = prev_output_tokens.ne(self.pad_idx).int()

        # set position ids to exclude padding symbols
        position_ids = attention_mask * (
            torch.arange(0, prev_output_tokens.size(1))
            .to(prev_output_tokens)
            .repeat(prev_output_tokens.size(0), 1)
        )

        outputs = self.gpt2.transformer(
            input_ids=prev_output_tokens,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        word_representations = outputs[0]
        #word_representations = word_representations[:, 1:, :]

        batch_size = word_representations.size(0)
        max_len = word_representations.size(1)
        scale = word_representations.size(2)**0.5

        scores = torch.matmul(word_representations, self.V).squeeze(-1)
        scores = scores / scale
        #Mask the padding values
        mask = torch.zeros(batch_size, max_len).to(self.device)
        for i in range(batch_size):
            mask[i, seq_lengths[i]:] = 1 #seq_lengths不考虑CLS和SEP的长度,CLS已去除，SEP和PAD都被mask掉
        scores = scores.masked_fill(mask.bool(), -np.inf)
        #Softmax, batch_size*1*max_len
        attn = self.softmax(scores).unsqueeze(1)
        #weighted sum, batch_size*hidden_dim. Sentence representations
        final_vec = torch.bmm(attn, word_representations).squeeze(1)
        final_vec = self.dropout(final_vec)
        #instance-level polarity score
        outputs = self.classifier(final_vec)

        return outputs


    def extract_condition_vecs(self,word_representations,seq_lengths):
        #word_representations = word_representations[:, 1:, :]
        batch_size = word_representations.size(0)
        max_len = word_representations.size(1)
        scale = word_representations.size(2)**0.5

        scores = torch.matmul(word_representations, self.V).squeeze(-1)
        scores = scores / scale
        #Mask the padding values
        mask = torch.zeros(batch_size, max_len).to(self.device)
        for i in range(batch_size):
            mask[i, seq_lengths[i]:] = 1 #seq_lengths不考虑CLS和SEP的长度,CLS已去除，SEP和PAD都被mask掉
        scores = scores.masked_fill(mask.bool(), -np.inf)
        #Softmax, batch_size*1*max_len
        attn = self.softmax(scores).unsqueeze(1)
        #weighted sum, batch_size*hidden_dim. Sentence representations
        final_vec = torch.bmm(attn, word_representations).squeeze(1)
        final_vec = self.dropout(final_vec)
        #instance-level polarity score
        outputs = self.classifier(final_vec)

        return outputs
    def max_positions(self):
        return self.model.config.n_positions - 1


class SentLevelGPT2Classifier(nn.Module):
    def __init__(self, args):
        super(SentLevelGPT2Classifier, self).__init__()
        config = GPT2Config.from_pretrained(args.pretrained_model_path)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(args.pretrained_model_path)
        #self.device = args.device
        self.num_labels = args.num_labels

        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)

        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        # set zero embedding for padding symbol
        self.pad_idx = 50256
        self.eos_idx = 2
        self.bos_idx = 2
        self.sent_id = 0

    def forward(
        self,
        prev_output_tokens,
        incremental_state = None,
        seq_lengths= None,
        return_feature=True,
    ):
        if incremental_state:
            past_key_values = self.get_incremental_state("past_key_values")
        else:
            past_key_values = None

        # don't attend to padding symbols
        attention_mask = prev_output_tokens.ne(self.pad_idx).int()

        # set position ids to exclude padding symbols
        position_ids = attention_mask * (
            torch.arange(0, prev_output_tokens.size(1))
            .to(prev_output_tokens)
            .repeat(prev_output_tokens.size(0), 1)
        )

        outputs = self.gpt2.transformer(
            input_ids=prev_output_tokens,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        word_representation = outputs[0]
        features,return_features = [],[]
        for index,batch in enumerate(word_representation):
            #features.append(batch[-seq_lengths[index],:])
            features.append(batch[-1,:]) #use the last token's rep to perform attribute classification
            if return_feature:
                return_features.append(batch[-2,:]) # return the second to last token's rep for retrieval
        features = torch.stack(features).to(prev_output_tokens.device)
        features = self.dropout(features)
        outputs = self.classifier(features)
        if return_feature:
            return_features = torch.stack(return_features).to(prev_output_tokens.device)
            return return_features, outputs
        else:
            return  outputs

    def extract_feature(self,
            prev_output_tokens,
            incremental_state=None,
            seq_lengths=None
            ):
        if incremental_state:
            past_key_values = self.get_incremental_state("past_key_values")
        else:
            past_key_values = None

        attention_mask = prev_output_tokens.ne(self.pad_idx).int()

        position_ids = attention_mask * (
            torch.arange(0, prev_output_tokens.size(1))
                .to(prev_output_tokens)
                .repeat(prev_output_tokens.size(0), 1)
        )

        outputs = self.gpt2.transformer(
            input_ids=prev_output_tokens,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        word_representations = outputs[0]


        # when batch-size ==1
        features = word_representations.squeeze(0)
        sent_ids = torch.full((1, seq_lengths[0]), self.sent_id)
        self.sent_id += 1

        # when batch-size >1
        # features, sent_ids = [], []
        # for index, batch  in enumerate(word_representations):
        #     features.append(batch[:seq_lengths[index], :])
        #     sent_ids.append(torch.full((1,seq_lengths[index]),self.sent_id))
        #     self.sent_id+=1
        # features = torch.cat(features,dim=0)
        # sent_ids = torch.cat(sent_ids,dim=-1).squeeze(0)
        return features, sent_ids

    def extract_condition_vecs(self, word_representation):
        word_representation = self.dropout(word_representation)
        features = self.classifier(word_representation)
        return features