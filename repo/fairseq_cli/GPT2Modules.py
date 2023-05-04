import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model
import numpy as np

class GPT2Classifier(nn.Module):
    def __init__(self, args):
        super(GPT2Classifier, self).__init__()
        config = GPT2Config.from_pretrained(r"G:\Projects\KNNLM\models\gpt2-medium")
        self.gpt2 = GPT2LMHeadModel.from_pretrained(r"G:\Projects\KNNLM\models\gpt2-medium")
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

    def max_positions(self):
        return self.model.config.n_positions - 1
