import torch
import faiss
import math
import numpy as np
from fairseq import utils
import time
from fairseq.data import Dictionary

class KNN_Dstore(object):
    def __init__(self, args):
        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.k = args.k
        self.dstore_size = args.dstore_size
        self.metric_type = args.faiss_metric_type
        self.sim_func = args.knn_sim_func
        self.dstore_fp16 = args.dstore_fp16
        self.index, self.condition_index = self.setup_faiss(args)
        self.n = float(args.similar_condition_prob)
        assert self.n <= self.k, "similar condition number must smaller than k number"


    def  setup_faiss(self, args):
        if not args.dstore_filename:
            raise ValueError('Cannot build a datastore without the data.')

        #start = time.time()
        index = faiss.read_index(args.indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)
        if args.topic_control:
            condition_index = faiss.read_index(args.indexfile.replace(".freezed","")+".topic", faiss.IO_FLAG_ONDISK_SAME_DIR)
        elif args.sentiment_control:
            condition_index = faiss.read_index(args.indexfile.replace(".freezed","")+".sentiment", faiss.IO_FLAG_ONDISK_SAME_DIR)

        #print('Reading datastore took {} s'.format(time.time() - start))
        index.nprobe = args.probe
        condition_index.nprobe = args.probe

        if args.dstore_fp16:
            print('Keys are fp16 and vals are int16')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int16, mode='r', shape=(self.dstore_size, 1))
            if args.sentiment_control:
                self.sent_ids = np.memmap(args.dstore_filename + '_sent_ids.npy', dtype=np.int16, mode='r',
                                      shape=(self.dstore_size, 1))


        else:
            #print('Keys are fp32 and vals are int64')
            if not args.no_load_keys:
                self.keys = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
            self.vals = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))
            if args.sentiment_control:
                self.sent_ids = np.memmap(args.dstore_filename.replace("_freezed_gpt2","") + '_sentiment_sent_ids.npy', dtype=np.int, mode='r',
                                      shape=(self.dstore_size, 1))
                self.condition_vectors = np.memmap(args.dstore_filename.replace("_freezed_gpt2","") + '_sentiment_vectors.npy', dtype=np.float32, mode='r',
                                      shape=(self.dstore_size, 2))
                self.features = np.memmap(args.dstore_filename.replace("_freezed_gpt2","") + '_gpt2_sentiment_features.npy', dtype=np.float32, mode='r',
                                          shape=(self.dstore_size, 1024))
            if args.topic_control:
                self.sent_ids = np.memmap(args.dstore_filename.replace("_freezed_gpt2","") + '_topic_sent_ids.npy', dtype=np.int, mode='r',
                                      shape=(self.dstore_size, 1))
                self.condition_vectors = np.memmap(args.dstore_filename.replace("_freezed_gpt2","") + '_topic_vectors.npy', dtype=np.float32, mode='r',
                                      shape=(self.dstore_size, 4))
                self.features = np.memmap(args.dstore_filename.replace("_freezed_gpt2","") + '_gpt2_topic_features.npy', dtype=np.float32, mode='r',
                                          shape=(self.dstore_size, 1024))

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if args.move_dstore_to_mem:
            print('Loading to memory...')
            start = time.time()

            if not args.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(args.dstore_filename+'_keys.npy', dtype=np.float32, mode='r', shape=(self.dstore_size, self.dimension))
                self.keys = np.zeros((self.dstore_size, self.dimension), dtype=np.float16 if args.dstore_fp16 else np.float32)
                self.keys = self.keys_from_memmap[:]
                self.keys = self.keys.astype(np.float16 if args.dstore_fp16 else np.float32)

            del self.vals
            self.vals_from_memmap = np.memmap(args.dstore_filename+'_vals.npy', dtype=np.int, mode='r', shape=(self.dstore_size, 1))
            self.vals = np.zeros((self.dstore_size, 1), dtype=np.int16 if args.dstore_fp16 else np.int)
            self.vals = self.vals_from_memmap[:]
            self.vals = self.vals.astype(np.int16 if args.dstore_fp16 else np.int)
            print('Loading to memory took {} s'.format(time.time() - start))

        return index, condition_index


    def get_knns(self, queries):
        start = time.time()
        dists, knns = self.index.search(queries.detach().cpu().float().numpy(), self.k)
        return dists, knns

    def get_condition_knns(self, queries):
        start = time.time()
        dists, knns = self.condition_index.search(queries.detach().cpu().float().numpy(), self.k)
        return dists, knns

    def get_knn_log_prob(self, queries, tgt, pad_idx):
        def dist_func(d, k, q, function=None):
            if not function:
                # Default behavior for L2 metric is to recompute distances.
                # Default behavior for IP metric is to return faiss distances.
                qsize = q.shape
                if self.metric_type == 'l2':
                    start = time.time()
                    knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
                    if self.half:
                        knns_vecs = knns_vecs.half()
                    query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                    l2 = torch.sum((query_vecs - knns_vecs.detach())**2, dim=2)
                    return -1 * l2
                return d

            if function == 'dot':
                qsize = q.shape
                return (torch.from_numpy(self.keys[k]).cuda() * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

            if function == 'do_not_recomp_l2':
                return -1 * d

            raise ValueError("Invalid knn similarity function!")

        # queries  are TxBxC
        # reshape: (TxB)xC
        qshape = queries.shape
        queries = queries.contiguous().view(-1, qshape[-1]) #token-num = queries[0] * queries[1]
        tgt = tgt.contiguous().view(-1)
        dists, knns = self.get_knns(queries[tgt != pad_idx]) #dists & knns: [token-num, k]
        # (T_reducedxB)xK
        dists = torch.from_numpy(dists).cuda()
        start = time.time()
        dists = dist_func(dists, knns, queries[tgt != pad_idx, :], function=self.sim_func) #dists = -1 * dists
        probs = utils.log_softmax(dists, dim=-1) #probs: [token-num, k]

        temp1 = torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1)
        temp2 = tgt[tgt != pad_idx].unsqueeze(-1)
        index_mask = torch.eq(temp1, temp2).float()
        index_mask[index_mask == 0] = -10000 # for stability
        index_mask[index_mask == 1] = 0 # #probs: [token-num, k]

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1).clone() # yhat_knn_prob: token-num
        full_yhat_knn_prob = torch.full([qshape[0]*qshape[1]], -10000).cuda().float()
        full_yhat_knn_prob[tgt != pad_idx] = yhat_knn_prob

        # TxBx1
        return full_yhat_knn_prob.view(qshape[0], qshape[1], 1)

