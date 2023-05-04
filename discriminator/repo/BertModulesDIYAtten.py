import torch
import torch.nn as nn
from pytorch_transformers import BertModel
import numpy as np

class BertClassifierDIYAtten(nn.Module):
    def __init__(self, config,args):
        super(BertClassifierDIYAtten, self).__init__()
        # Binary classification problem (num_labels = 2)
        self.device = args.device
        self.num_labels = config.num_labels
        # Pre-trained BERT model
        #self.bert = BertModel(config)
        self.bert = BertModel.from_pretrained(args.pretrained_model_path)
        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.V = nn.Parameter(torch.randn(config.hidden_size, 1))

        self.softmax = nn.Softmax(dim=-1)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Weight initialization
        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None,seq_lengths=None):
        # Forward pass through pre-trained BERT
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,attention_mask=attention_mask, head_mask=head_mask)

        # Last layer output (Total 12 layers)
        word_representations = outputs[0]
        word_representations = word_representations[:,1:,:] #去除CLS

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

    def prob(self, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None):

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,attention_mask=attention_mask, head_mask=head_mask)
        pooled_output = outputs[1]
        polarity = self.classifier(pooled_output)
        return outputs, polarity
                                  
