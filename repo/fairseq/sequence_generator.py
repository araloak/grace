# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import numpy as np
import faiss
from torch import nn, optim

from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.knnlm import KNN_Dstore
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
import os
#os.environ ['CUDA_VISIBLE_DEVICES'] = '1'
from .GPT2Modules import SentLevelGPT2Classifier
class SequenceGenerator(object):
    def __init__(
        self,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.,
        unk_penalty=0.,
        retain_dropout=False,
        temperature=1.,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        args=None
    ):
        """Generates translations of a given source sentence.

        Args:
            tgt_dict (~fairseq.data.Dictionary): target dictionary
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        assert temperature > 0, '--temperature must be greater than 0'

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )
        # my
        if "gpt2" in args.arch:
            self.vocab_size = 50257
            self.pad = 50256
            self.unk=50255
        self.args = args
        if self.args.sentiment_control:
            self.control_codes = np.array([line.strip() for line in open(args.prompt_control_code,encoding="utf8").readlines()])
        elif self.args.topic_control:
            self.control_codes = np.array([line.strip() for line in open(args.prompt_control_code,encoding="utf8").readlines()])

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        model = EnsembleModel(models,self.args)
        return self._generate(model, sample, **kwargs)

    @torch.no_grad()
    def _generate(
        self,
        model,
        sample,
        prefix_tokens=None,
        bos_token=None,
        **kwargs
    ):
        if not self.retain_dropout:
            model.eval()

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }

        src_tokens = encoder_input['src_tokens']
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        src_len = input_size[1]
        beam_size = self.beam_size

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )
        assert self.min_len <= max_len, 'min_len cannot be larger than max_len, please adjust these!'

        # compute the encoder output for each beam
        encoder_outs = model.forward_encoder(encoder_input)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = model.reorder_encoder_out(encoder_outs, new_order)

        # initialize buffers
        scores = src_tokens.new(bsz * beam_size, max_len + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.new(bsz * beam_size, max_len + 2).long().fill_(self.pad)
        tokens_buf = tokens.clone()
        if self.args.gpt2_gen_mode:# my
            # torch.index_select(
            #     tokens[:, :1], dim=0, index=active_bbsz_idx,
            #     out=tokens_buf[:, :step + 1],
            # )
            first_tokens = src_tokens[:, 0].repeat(beam_size,1).T.flatten() #my
            tokens[:, 0] =  first_tokens #my
        else:
            tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn, attn_buf = None, None

        # The blacklist indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then the blacklist would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        blacklist = src_tokens.new_zeros(bsz, beam_size).eq(-1)  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfin_idx):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size or step == max_len:
                return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            #tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            assert not tokens_clone.eq(self.eos).any()
            tokens_clone[:, step] = self.eos
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step+2] if attn is not None else None

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step+1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                if self.match_source_len and step > src_lengths[unfin_idx]:
                    score = -math.inf

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i]
                    else:
                        hypo_attn = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': None,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                #! check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfin_idx):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        reorder_state = None
        batch_idxs = None

        batch_size = sample['id'].size(0)
        beam_size = tokens.size(0) // batch_size
        if self.args.sentiment_control:
            positive_vec = torch.tensor([[0., 1.]]).to(src_tokens.device)
            negative_vec = torch.tensor([[1., 0.]]).to(src_tokens.device)
            # final_knn_num = self.dstore.n
            desired_condition_vectors, undesired_condition_vectors = [], []
            control_codes = self.control_codes[sample['id'].cpu()]

            # prepare control sentiment vectors
            if batch_size == 1:
                control_codes = [control_codes]
            for control_code in control_codes:
                if control_code == "positive":
                    desired_condition_vectors.append(torch.repeat_interleave(positive_vec, repeats=beam_size, dim=0))
                    undesired_condition_vectors.append(
                        torch.stack([torch.repeat_interleave(negative_vec, repeats=beam_size, dim=0)]))
                elif control_code == "negative":
                    desired_condition_vectors.append(torch.repeat_interleave(negative_vec, repeats=beam_size, dim=0))
                    undesired_condition_vectors.append(
                        torch.stack([torch.repeat_interleave(positive_vec, repeats=beam_size, dim=0)]))
            desired_condition_vectors = torch.stack(desired_condition_vectors, dim=0).to(src_tokens.device).view(
                batch_size * beam_size, -1)
            undesired_condition_vectors = torch.stack(undesired_condition_vectors, dim=0).to(src_tokens.device).view(
                batch_size * beam_size, 1, 2)

        if self.args.topic_control:
            world_vec = torch.tensor([[1., 0., 0., 0.]]).to(src_tokens.device)
            sports_vec = torch.tensor([[0., 1., 0., 0.]]).to(src_tokens.device)
            business_vec = torch.tensor([[0., 0., 1., 0.]]).to(src_tokens.device)
            tech_vec = torch.tensor([[0., 0., 0., 1.]]).to(src_tokens.device)
            desired_condition_vectors, undesired_condition_vectors = [], []
            control_codes = self.control_codes[sample['id'].cpu()]

            # prepare control sentiment vectors
            if batch_size == 1:
                control_codes = [control_codes]
            for control_code in control_codes:
                if control_code == "world":
                    desired_condition_vectors.append(torch.repeat_interleave(world_vec, repeats=beam_size, dim=0))
                    undesired_condition_vectors.append(torch.stack(
                        [torch.repeat_interleave(sports_vec, repeats=beam_size, dim=0),
                         torch.repeat_interleave(business_vec, repeats=beam_size, dim=0),
                         torch.repeat_interleave(tech_vec, repeats=beam_size, dim=0)]))
                elif control_code == "sports":
                    desired_condition_vectors.append(torch.repeat_interleave(sports_vec, repeats=beam_size, dim=0))
                    undesired_condition_vectors.append(torch.stack(
                        [
                         torch.repeat_interleave(world_vec, repeats=beam_size, dim=0),
                         torch.repeat_interleave(business_vec, repeats=beam_size, dim=0),
                         torch.repeat_interleave(tech_vec, repeats=beam_size, dim=0)
                        ]))
                elif control_code == "business":
                    desired_condition_vectors.append(torch.repeat_interleave(business_vec, repeats=beam_size, dim=0))
                    undesired_condition_vectors.append(torch.stack(
                        [torch.repeat_interleave(world_vec, repeats=beam_size, dim=0),
                         torch.repeat_interleave(sports_vec, repeats=beam_size, dim=0),
                         torch.repeat_interleave(tech_vec, repeats=beam_size, dim=0)]))
                elif control_code == "tech":
                    desired_condition_vectors.append(torch.repeat_interleave(tech_vec, repeats=beam_size, dim=0))
                    undesired_condition_vectors.append(torch.stack(
                        [torch.repeat_interleave(world_vec, repeats=beam_size, dim=0),
                         torch.repeat_interleave(business_vec, repeats=beam_size, dim=0),
                         torch.repeat_interleave(sports_vec, repeats=beam_size, dim=0)]))
            desired_condition_vectors = torch.stack(desired_condition_vectors, dim=0).view(batch_size * beam_size, -1)
            undesired_condition_vectors = torch.stack(undesired_condition_vectors, dim=0).view(batch_size * beam_size, 3, 4) #multi-attribute

        if not self.args.topic_control and not self.args.sentiment_control:
            desired_condition_vectors = None
            undesired_condition_vectors = None
        step=0

        will_refine = False
        before_replaced_tokens, before_refined_token = None, None
        before_replaced_tokens_buf, before_refined_tokens_buf = None, None
        # token_replaced = False
        while step <= max_len:
        #for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                model.reorder_incremental_state(reorder_state)
                encoder_outs = model.reorder_encoder_out(encoder_outs, reorder_state)

            if self.args.refine and will_refine:
                will_refine = False
                inputs = {
                    'prev_output_tokens': tokens[:, :step + 2]
                }
                labels = torch.topk(desired_condition_vectors, 1)[1].squeeze(1)
                _, logits = model.init_classifier(**inputs)
                _, predicted = torch.max(logits.data, 1)
                #refined_indice = torch.nonzero(labels != predicted).squeeze(1)
                refined_indice = torch.nonzero(labels == predicted).squeeze(1)  # gradiant-analyze
                if refined_indice.shape[0] !=0:
                    before_refined_token = tokens
                    before_refined_tokens_buf = tokens_buf
                    tokens = torch.full_like(tokens,self.pad)
                    tokens_buf = torch.full_like(tokens_buf,self.pad)
                    tokens[refined_indice] = before_refined_token[refined_indice]
                    tokens_buf[refined_indice] = before_refined_tokens_buf[refined_indice]
                    lprobs, avg_attn_scores = model.forward_decoder(
                        tokens[:, :step + 2], encoder_outs, temperature=self.temperature, sample_ids=sample['id'], desired_condition_vectors=desired_condition_vectors, undesired_condition_vectors =undesired_condition_vectors,refine=True
                    )
                    refined = True
                else:
                    refined = False
                    next_step = 1
                    step += next_step
                    continue
                next_step = 1

            else:
                if before_refined_token is not None and self.args.refine:
                    #labels = torch.topk(desired_condition_vectors, 1)[1].squeeze(1)
                    _, refined_logits = model.init_classifier(tokens[:, :step + 1])
                    #_, refined_predicted = torch.max(refined_logits.data, 1)
                    refined_simi_score = torch.cosine_similarity(refined_logits, desired_condition_vectors, dim=-1)
                    _, previous_logits = model.init_classifier(before_refined_token[:, :step + 1])
                    #_, previous_predicted = torch.max(previous_logits.data, 1)
                    previous_simi_score = torch.cosine_similarity(previous_logits, desired_condition_vectors,dim=-1)
                    better_indice = torch.gt(refined_simi_score, previous_simi_score)
                    better_indice = torch.nonzero(better_indice).squeeze(1)
                    replace_indice = []
                    for id in better_indice:
                        if torch.eq(refined_indice,id).any():
                            replace_indice.append(id)
                    if replace_indice != []:
                        replace_indice = torch.stack(replace_indice)
                        before_refined_token[replace_indice] = tokens[replace_indice]
                        before_refined_tokens_buf[replace_indice] = tokens_buf[replace_indice]
                    tokens = before_refined_token
                    tokens_buf = before_refined_tokens_buf
                    before_refined_token = None

                lprobs, avg_attn_scores = model.forward_decoder(
                    tokens[:, :step + 1], encoder_outs, temperature=self.temperature, sample_ids=sample['id'],
                desired_condition_vectors=desired_condition_vectors, undesired_condition_vectors =undesired_condition_vectors,refine=False)
                if self.args.refine:
                    next_step = 0
                else:
                    next_step = 1
                will_refine = True
                refined = False
            # if self.args.refine and refined:
            #     previous_tokens = tokens
            #     previous_tokens_buf = tokens_buf

            lprobs[lprobs != lprobs] = -math.inf

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if prefix_tokens is not None and step < prefix_tokens.size(1) and step < max_len:
                prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
                prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
                prefix_mask = prefix_toks.ne(self.pad)
                lprobs[prefix_mask] = -math.inf
                lprobs[prefix_mask] = lprobs[prefix_mask].scatter_(
                    -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
                )
                # if prefix includes eos, then we should make sure tokens and
                # scores are the same across all beams
                eos_mask = prefix_toks.eq(self.eos)
                if eos_mask.any():
                    # validate that the first beam matches the prefix
                    first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[:, 0, 1:step + 1]
                    eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
                    target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
                    assert (first_beam == target_prefix).all()

                    def replicate_first_beam(tensor, mask):
                        tensor = tensor.view(-1, beam_size, tensor.size(-1))
                        tensor[mask] = tensor[mask][:, :1, :]
                        return tensor.view(-1, tensor.size(-1))

                    # copy tokens, scores and lprobs from the first beam to all beams
                    tokens = replicate_first_beam(tokens, eos_mask_batch_dim)
                    scores = replicate_first_beam(scores, eos_mask_batch_dim)
                    lprobs = replicate_first_beam(lprobs, eos_mask_batch_dim)
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            if self.no_repeat_ngram_size > 0:
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for bbsz_idx in range(bsz * beam_size)]
                for bbsz_idx in range(bsz * beam_size):
                    gen_tokens = tokens[bbsz_idx].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                                gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            # Record attention scores
            if type(avg_attn_scores) is list:
                avg_attn_scores = avg_attn_scores[0]
            if avg_attn_scores is not None:
                if attn is None:
                    attn = scores.new(bsz * beam_size, src_tokens.size(1), max_len + 2)
                    attn_buf = attn.clone()
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)

            self.search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:
                def calculate_banned_tokens(bbsz_idx):
                    # before decoding the next token, prevent decoding of ngrams that have already appeared
                    ngram_index = tuple(tokens[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
                    return gen_ngrams[bbsz_idx].get(ngram_index, [])

                if step + 2 - self.no_repeat_ngram_size >= 0:
                    # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                    banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(bsz * beam_size)]
                else:
                    banned_tokens = [[] for bbsz_idx in range(bsz * beam_size)]

                for bbsz_idx in range(bsz * beam_size):
                    lprobs[bbsz_idx, banned_tokens[bbsz_idx]] = -math.inf

            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos, except for blacklisted ones
            # or candidates with a score of -inf
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][blacklist] = 0

            # only consider eos when it's among the top beam_size indices
            torch.masked_select(
                cand_bbsz_idx[:, :beam_size],
                mask=eos_mask[:, :beam_size],
                out=eos_bbsz_idx,
            )

            finalized_sents = set()
            if eos_bbsz_idx.numel() > 0:
                torch.masked_select(
                    cand_scores[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_scores,
                )
                finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores)
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < max_len

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                blacklist = blacklist[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                    attn_buf.resize_as_(attn)
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos or
            # blacklisted hypos and values < cand_size indicate candidate
            # active hypos. After this, the min values per row are the top
            # candidate active hypos.
            active_mask = buffer('active_mask')
            eos_mask[:, :beam_size] |= blacklist
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, new_blacklist = buffer('active_hypos'), buffer('new_blacklist')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(new_blacklist, active_hypos)
            )

            # update blacklist to ignore any finalized hypos
            blacklist = new_blacklist.ge(cand_size)[:, :beam_size]
            assert (~blacklist).any(dim=1).all()

            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],
            )
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                    out=attn_buf[:, :, :step + 2],
                )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx
            step+=next_step


        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)
        return finalized


class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models, args=None):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None
        if all(hasattr(m, 'decoder') and isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.incremental_states = {m: {} for m in models}

        self.args = args #my
        if "gpt2" in args.arch:
            self.pad = 50256
        else:
            self.pad = 1
        # DIY
        if self.args.knnlm:
            self.args.knn_keytype = "last_ffn_input"
            if "gpt2_medium" in self.args.arch:
                self.args.decoder_embed_dim = 1024
            self.dstore = KNN_Dstore(self.args)
        else:
            self.dstore = None
        if args.refine or True:
            if args.topic_control:
                args.num_labels = 4
            elif args.sentiment_control:
                args.num_labels = 2
            args.device = "cuda" if not args.cpu else "cpu"
            self.classifier = SentLevelGPT2Classifier(args).to(args.device)
            self.init_classifier = SentLevelGPT2Classifier(args).to(args.device)
            state_dict = torch.load(args.classifier_path + "/best_model.pt")
            self.classifier.load_state_dict(state_dict)#.to(models.device)
            self.init_classifier.load_state_dict(state_dict)
            self.rep_store = torch.zeros((5000,3,1024)).to(args.device).float()
            self.rep_num = 0


    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        return [model.encoder(**encoder_input) for model in self.models]

    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_outs, temperature=1., sample_ids=None, desired_condition_vectors=None, undesired_condition_vectors =None,refine=None):
        if len(self.models) == 1:
            return self._decode_one(
                tokens,
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
                sample_ids=sample_ids, # my
                desired_condition_vectors=desired_condition_vectors,
                undesired_condition_vectors=undesired_condition_vectors,
                refine=refine
            )

        log_probs = []
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn = self._decode_one(
                tokens,
                model,
                encoder_out,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1., sample_ids=None, desired_condition_vectors=None, undesired_condition_vectors=None, refine=None, **kwargs
    ):

        def decay_rate(step, bar=40, init = 0.8,low=0.):
            #lmbda最开始设为0.8，随着step变大逐步递减，最小减小为step>=40时为0.1
            if step >bar:
                rate = low
            else:
                rate = (low - init)/bar + init

            return rate

        if refine : #test
            labels = torch.topk(desired_condition_vectors, 1)[1].squeeze(1)
            inputs = {
                'prev_output_tokens': tokens,
            }
            self.classifier.train()
            optimizer = torch.optim.Adam([
                {'params': self.classifier.gpt2.parameters(), 'lr': 1e-5},
                {'params': self.classifier.classifier.parameters(), 'lr': 3e-4},
            ])
            optimizer.zero_grad()
            criterion = nn.CrossEntropyLoss()
            original_disc_feature, predictions = self.classifier(**inputs)
            loss = criterion(predictions, labels)
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            self.classifier.eval()
            features, predictions = self.classifier(**inputs)
            tokens = tokens[:, :-1]
            self.classifier = self.init_classifier
            decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out, args=self.args))
            decoder_out[0] = decoder_out[0][:, -1:, :]
            attn = decoder_out[1]
            plm_rep = decoder_out[1]['last_ffn_input'].squeeze(1)[-1].unsqueeze(0)
            self.rep_store[self.rep_num:self.rep_num+plm_rep.size(0),0,:] = plm_rep
            self.rep_store[self.rep_num:self.rep_num+plm_rep.size(0),1,:] = original_disc_feature
            self.rep_store[self.rep_num:self.rep_num+plm_rep.size(0),2,:] = features
            self.rep_num+=plm_rep.size(0)
            if type(attn) is dict:
                attn = attn.get('attn', None)
            if type(attn) is list:
                attn = attn[0]
            if attn is not None:
                attn = attn[:, -1, :]
            #probs = torch.nn.functional.softmax(decoder_out[0],dim=-1)[:, -1, :]
             # .squeeze(0)

        else:
            if self.incremental_states is not None:
                decoder_out = list(model.forward_decoder(
                    tokens, encoder_out=encoder_out, incremental_state=self.incremental_states[model], args=self.args
                ))
            else:
                decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out))
            decoder_out[0] = decoder_out[0][:, -1:, :]
            if temperature != 1.:
                decoder_out[0].div_(temperature)
            attn = decoder_out[1]
            if type(attn) is dict:
                attn = attn.get('attn', None)
            if type(attn) is list:
                attn = attn[0]
            if attn is not None:
                attn = attn[:, -1, :]
        probs = torch.nn.functional.softmax(decoder_out[0],dim=-1)[:, -1, :]  # batch_size * vocab_len

            # when gen with only gpt2-medium
            # probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
            # probs = probs[:, -1, :]

        if self.args.knnlm:
            if refine:
                queries = features.contiguous()
                #knns = np.random.randint(0, 8557168, (queries.size(0), 1000))
                dists, knns = self.dstore.get_condition_knns(queries)
                # temp1 = torch.from_numpy(dstore.vals[knns]).long().cuda().squeeze(-1)
                # for i,each in enumerate(temp1):
                #     results = tokenizer.decode(each)
                #     tem = tokenizer.decode(tokens[i].squeeze(-1))
                #     print(results)
            else:
                queries = decoder_out[1][self.args.knn_keytype][-1, :, :].contiguous()

                labels = torch.topk(desired_condition_vectors, 1)[1].squeeze(1)
                inputs = {
                    'prev_output_tokens': tokens,
                    'return_feature': False,
                }
                self.classifier.eval()
                logits = self.classifier(**inputs)
                _, predicted = torch.max(logits.data, 1)
                retrieve_indice = torch.nonzero(labels == labels).squeeze(1)
                #retrieve_indice = torch.nonzero(labels != predicted).squeeze(1)

                if len(retrieve_indice) == 0 or tokens.shape[1] > self.args.max_control_step:
                    probs = torch.log(probs)
                    return probs, attn
                queries = queries[retrieve_indice]
                dists, knns = self.dstore.get_knns(queries)

                #knns = np.random.randint(0, 8557168, (queries.size(0), 1000))

                #temp1 = torch.from_numpy(self.dstore.vals[knns]).long().cuda().squeeze(-1)
                #for i,each in enumerate(temp1):
                    #results = tokenizer.decode(each)

            if self.args.faiss_metric_type == 'l2':
                targets = torch.from_numpy(self.dstore.vals[knns]).long().cuda().squeeze(-1)#.squeeze(0)
                features = torch.from_numpy(self.dstore.features[knns]).to(probs.device).view(queries.shape[0], self.dstore.k, -1)
                #features = torch.from_numpy(self.dstore.keys[knns]).to(probs.device).view(queries.shape[0], self.dstore.k, -1)
            # batch_size = sample_ids.size(0)
            # beam_size = queries.size(0)//sample_ids.size(0)
            vocab_size = probs.shape[1]

           # use knn id search for subseq condition vectors
            if self.args.topic_control or self.args.sentiment_control:
                #sent_ids = self.dstore.sent_ids[knns]
                condition_vectors = torch.from_numpy(self.dstore.condition_vectors[knns]).to(queries.device)

                # cal cosine simi of desired and undesired conditions
                sentiment_similarities, undesired_sentiment_similarities = [], []
                #print(undesired_condition_vectors, desired_condition_vectors)

                for undesired_condition_vector_group, desired_condition_vector,condition_vector in zip(undesired_condition_vectors, desired_condition_vectors,condition_vectors):
                    desired_condition_vector = desired_condition_vector.unsqueeze(0)
                    sentiment_similarities.append(torch.cosine_similarity(desired_condition_vector.unsqueeze(1), condition_vector.unsqueeze(0),dim=-1))
                    #sentiment_similarities.append(torch.cosine_similarity(torch.tensor([[0.,0.,0.,1]]).to(queries.device).unsqueeze(1), condition_vector.unsqueeze(0),dim=-1)) #multiattribue
                    undesired_sentiment_group_similarities = []

                    for undesired_condition_vector in undesired_condition_vector_group:
                        undesired_condition_vector = undesired_condition_vector.unsqueeze(0)
                        undesired_sentiment_group_similarities.append(torch.cosine_similarity(undesired_condition_vector.unsqueeze(1),condition_vector.unsqueeze(0), dim=-1))
                    undesired_sentiment_similarities.append(torch.stack(undesired_sentiment_group_similarities))

                sentiment_similarities = torch.stack(sentiment_similarities,dim=0).to(queries.device).squeeze(1)
                undesired_sentiment_similarities = torch.stack(undesired_sentiment_similarities,dim=0).to(queries.device).squeeze(1)

                batch_combined_knn_p, batch_undesired_combined_knn_p = [], []
                for i, (batch_simi, undesired_batch_simi_group, feature,target) in enumerate(zip(sentiment_similarities,undesired_sentiment_similarities, features,targets)):
                    indice = torch.nonzero(batch_simi > self.dstore.n, as_tuple=False).squeeze(1)
                    undesired_indice = []
                    for undesired_batch_simi in undesired_batch_simi_group:
                        undesired_indice.append(torch.nonzero(undesired_batch_simi.squeeze(0) > self.dstore.n, as_tuple=False).squeeze(1))
                    undesired_indice = torch.cat(undesired_indice,dim=-1)
                    # desired_vec_num = indice.shape[0]
                    # undesired_vec_num = undesired_indice.shape[0]
                    condition_knns_vecs = feature[indice]
                    undesired_condition_knns_vecs = feature[undesired_indice]

                    # good_targets = target[indice]
                    # # bad_targets = targets[undesired_indice]
                    # good_results = tokenizer.decode(good_targets)
                    # #print(good_results)
                    # tem = tokenizer.decode(tokens[i].squeeze(-1))
                    # # print(tem)
                    # with open(r"G:\Projects\KNNLM\data\business-retrieve-words.txt", "a", encoding='utf8') as f:
                    #     f.write(tem.replace("\n", "") + "\n")
                    #     f.write(good_results+"\n")
                    # tem = tokenizer.decode(tokens[i].squeeze(-1))
                    # bad_results = tokenizer.decode(bad_targets)

                    with torch.no_grad():
                        #print(condition_knns_vecs.shape)
                        a = model.decoder.model.lm_head(condition_knns_vecs)
                        a = torch.nn.functional.softmax(a, dim=-1)

                        undesired_a = model.decoder.model.lm_head(undesired_condition_knns_vecs)
                        undesired_a = torch.nn.functional.softmax(undesired_a, dim=-1)

                    if a.shape[0] != 0:# 可能为空 tensor
                        combined_knn_p = torch.sum(a, dim=0) / a.shape[0]
                    else:
                        combined_knn_p = torch.zeros(vocab_size).to(queries.device).float()
                    if undesired_a.shape[0] != 0:
                        undesired_combined_knn_p = torch.sum(undesired_a, dim=0) / undesired_a.shape[0]
                    else:
                        undesired_combined_knn_p = torch.zeros(vocab_size).to(queries.device).float()

                    batch_combined_knn_p.append(combined_knn_p)
                    batch_undesired_combined_knn_p.append(undesired_combined_knn_p)

            batch_combined_knn_p = torch.stack(batch_combined_knn_p, dim=0)
            batch_undesired_combined_knn_p = torch.stack(batch_undesired_combined_knn_p, dim=0)

            # topic world/business control setting
            if "world" in self.args.prompt_control_code or "business" in self.args.prompt_control_code:
                rate = decay_rate(tokens.shape[1], bar=80,init=self.args.lmbda, low=self.args.lmbda-0.4)
            elif "tech" in self.args.prompt_control_code or "sports" in self.args.prompt_control_code:
                # topic tech/sports control setting
                rate = decay_rate(tokens.shape[1], bar=40,init=self.args.lmbda, low=self.args.lmbda-0.4)
            elif "negative" in self.args.prompt_control_code:
                rate = decay_rate(tokens.shape[1], bar=80, init=self.args.lmbda, low=self.args.min_lmbda)
            elif "positive" in self.args.prompt_control_code:
                rate = decay_rate(tokens.shape[1], bar=80, init=self.args.lmbda, low=self.args.min_lmbda)

            if refine:
                #probs = batch_combined_knn_p *  rate + probs * (1-rate) - batch_undesired_combined_knn_p * rate
                probs = batch_combined_knn_p * self.args.lmbda  + probs * (1-self.args.lmbda) - batch_undesired_combined_knn_p * self.args.lmbda

            else:
                # with open(r"G:\Projects\KNNLM\data\business-prob-words.txt", "a", encoding='utf8') as f:
                #     for i,layer in enumerate(batch_combined_knn_p):
                #         tem = tokenizer.decode(tokens[i].squeeze(-1))
                #         _,words = torch.topk(layer ,100)
                #         words = tokenizer.decode(words)
                #         f.write(tem.replace("\n", "") + "\n")
                #         f.write(words + "\n")
                probs[retrieve_indice] = batch_combined_knn_p *  rate + probs[retrieve_indice] * (1-rate) - batch_undesired_combined_knn_p * rate
                #probs[retrieve_indice] = batch_combined_knn_p * self.args.lmbda  + probs[retrieve_indice] * (1-self.args.lmbda) - batch_undesired_combined_knn_p * self.args.lmbda

            #probs = batch_combined_knn_p * self.args.lmbda  + probs * (1-self.args.lmbda) - batch_undesired_combined_knn_p * self.args.lmbda

            probs = torch.clamp(probs, min=1e-12)
            probs = torch.nn.functional.normalize(probs, p=1, dim=1)
        probs = torch.log(probs)



        return probs, attn

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]

    def reorder_incremental_state(self, new_order):
        if self.incremental_states is None:
            return
        for model in self.models:
            model.decoder.reorder_incremental_state(self.incremental_states[model], new_order)


class SequenceGeneratorWithAlignment(SequenceGenerator):

    def __init__(self, tgt_dict, left_pad_target=False, **kwargs):
        """Generates translations of a given source sentence.

        Produces alignments following "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            left_pad_target (bool, optional): Whether or not the
                hypothesis should be left padded or not when they are
                teacher forced for generating alignments.
        """
        super().__init__(tgt_dict, **kwargs)
        self.left_pad_target = left_pad_target

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        model = EnsembleModelWithAlignment(models)
        finalized = super()._generate(model, sample, **kwargs)

        src_tokens = sample['net_input']['src_tokens']
        bsz = src_tokens.shape[0]
        beam_size = self.beam_size
        src_tokens, src_lengths, prev_output_tokens, tgt_tokens = \
            self._prepare_batch_for_alignment(sample, finalized)
        if any(getattr(m, 'full_context_alignment', False) for m in model.models):
            attn = model.forward_align(src_tokens, src_lengths, prev_output_tokens)
        else:
            attn = [
                finalized[i // beam_size][i % beam_size]['attention'].transpose(1, 0)
                for i in range(bsz * beam_size)
            ]

        # Process the attn matrix to extract hard alignments.
        for i in range(bsz * beam_size):
            alignment = utils.extract_hard_alignment(attn[i], src_tokens[i], tgt_tokens[i], self.pad, self.eos)
            finalized[i // beam_size][i % beam_size]['alignment'] = alignment
        return finalized

    def _prepare_batch_for_alignment(self, sample, hypothesis):
        src_tokens = sample['net_input']['src_tokens']
        bsz = src_tokens.shape[0]
        src_tokens = src_tokens[:, None, :].expand(-1, self.beam_size, -1).contiguous().view(bsz * self.beam_size, -1)
        src_lengths = sample['net_input']['src_lengths']
        src_lengths = src_lengths[:, None].expand(-1, self.beam_size).contiguous().view(bsz * self.beam_size)
        prev_output_tokens = data_utils.collate_tokens(
            [beam['tokens'] for example in hypothesis for beam in example],
            self.pad, self.eos, self.left_pad_target, move_eos_to_beginning=True,
        )
        tgt_tokens = data_utils.collate_tokens(
            [beam['tokens'] for example in hypothesis for beam in example],
            self.pad, self.eos, self.left_pad_target, move_eos_to_beginning=False,
        )
        return src_tokens, src_lengths, prev_output_tokens, tgt_tokens


class EnsembleModelWithAlignment(EnsembleModel):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__(models)

    def forward_align(self, src_tokens, src_lengths, prev_output_tokens):
        avg_attn = None
        for model in self.models:
            decoder_out = model(src_tokens, src_lengths, prev_output_tokens)
            attn = decoder_out[1]['attn']
            if avg_attn is None:
                avg_attn = attn
            else:
                avg_attn.add_(attn)
        if len(self.models) > 1:
            avg_attn.div_(len(self.models))
        return avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1.,
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens,
                encoder_out=encoder_out,
                incremental_state=self.incremental_states[model],
            ))
        else:
            decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if type(attn) is list:
            attn = attn[0]
        if attn is not None:
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        return probs, attn
