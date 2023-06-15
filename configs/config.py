#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn

from utils.model_utils import xavier_weights_init, normal_weights_init

ROOT_DIR = "./"
DATA_DIR = f"{ROOT_DIR}data/"

follow = {
    "writer" : True,
    "Name" : "__lm_run00",
    "write_rec_loss" : True,
    "write_total_loss" : True,
    "write_ROUGE" : True
}


class DatasetConfig():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data paths
        self.data_dir = DATA_DIR
        self.train_data = f"{DATA_DIR}processed/train.min.csv"#20221204_amazon_reviews_train.csv"#train.half.csv"#
        self.valid_data = f"{DATA_DIR}processed/valid.min.csv"#20221204_amazon_reviews_valid.csv"#valid.half.csv"#
        self.test_data = f"{DATA_DIR}processed/20221204_amazon_reviews_test.csv"
        
        # Dataloader configs
        self.batch_size = 1
        self.vocab_size = 50000
        self.min_freq = 1
        self.truncate = False
        self.max_num_reviews = 8
        self.max_len_rev = None
        self.remove_punctuations = False
        self.remove_stopwords = False
        
        self.pin_memory = False
        self.workers = 1


class HP():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        ######## MODEL PARAMETTERS ########
        self.weights_init = normal_weights_init #xavier_weights_init
        self.use_rec = True # Enable text reconstruction. If False, only sentiment classification is done
        self.use_cls = True  # Specify if document (sentiment) classification should be done
        self.use_grl = True  # Specify if the gradient reversal layer should be used
        self.use_proj = True  # Specify if the projection mechanism should be used
        self.use_gat = False  # Specify in the Graph Attention NN (GAT) should be used in the encoder
        
        assert (not self.use_rec and self.use_cls) or self.use_rec
        
        # The GRL is only useful during classification
        assert (self.use_grl and self.use_cls) or (self.use_grl is False)
        # The projection mechanism can only be used when classification is enabled as well as grl
        assert (self.use_proj and self.use_cls and self.use_grl) or (self.use_proj is False)
        
        # Indicates the number of documents used as "target" documents.
        # 1 = a document of opposite domain (sentiment) as the input document
        # 2 = same as (1) but with an additional document of the same domain (sentiment) as the input document
        # 3 = same as (2) but with two documents of opposite domain (sentiment) as the input document
        # -1 = a concatenation of all documents having an opposite domain (sentiment) as the input document
        # all in the same batch
        self.num_tgr_docs_domains = 1
        self.concat_docs = False
        assert self.num_tgr_docs_domains in [0, 1, 2]
        
        # Word embedding dimensions
        self.enc_emb_dim = 256
        self.dec_emb_dim = 256
        self.emb_dim = 200
        self.use_pretrained_embs = True  # Defined whether a pretrained model should be used for word embeddings
        self.glove_name = "6B"
        
        if self.use_pretrained_embs:
            self.enc_emb_dim = self.emb_dim
            self.dec_emb_dim = self.emb_dim
        
        # Auto encoder
        self.hid_dim = 512
        self.enc_layers = 1
        self.enc_drop = 0.1
        self.dec_drop = 0.1
        # dec_hidden_type: defines which hidden state to use during the decoding phase
        # 0: regular hidden states (H)
        # 1: domain-independent hidden states (H_hat)
        # 2: domain-specific hidden states (H_tilt)
        self.dec_hidden_type = 2  # Used to train the language model (reconstruction)
        self.mean_hidden_type = 0 # Used to compute the mean representation of the input reviews (default H)
        self.mean_context_type = 0 # Used to compute the mean representation of the input reviews (default H)
        self.ref_hidden_type = 1  # Used to compute the cosine similarity (default H_hat)
        self.gen_hidden_type = 0  # Used to generate summaries during inference (default H)
        
        assert (self.dec_hidden_type in [0, 1, 2] and self.use_proj) or self.dec_hidden_type == 0
        assert (self.mean_hidden_type in [0, 1, 2] and self.use_proj) or self.mean_hidden_type == 0
        assert (self.mean_context_type in [0, 1, 2] and self.use_proj) or self.mean_context_type == 0
        assert (self.ref_hidden_type in [0, 1, 2] and self.use_proj) or self.ref_hidden_type == 0
        assert (self.gen_hidden_type in [0, 1, 2] and self.use_proj) or self.gen_hidden_type == 0
        
        # Summarizer
        self.combine_encs = "mean"
        
        # GAT
        self.gat_hidden = 200
        self.gat_heads = 2
        self.gat_alpha = 0.2
        self.gat_drop = 0.5
        
        # Classifier
        self.cls_hidden = 200
        self.cls_classes = 2
        
        # GRL
        self.grl_gamma = 10.0
        
        # Beam decoding
        self.beam_decode = False
        self.beam_size = 5 
        self.min_dec_steps = 5
        self.num_return_seq = 5 
        self.num_return_sum = 1
        self.n_gram_block = 3
        
        ######## TRAINING PARAMETTERS ########
        self.cls_num_epochs = 1 # Number of epochs for which train the classifier alone
        self.cls_weight = 1.0 # Classifier loss weight
        self.rec_weight = 1.0 # reconstruction loss weight
        self.acc_size = 64    # gradient accumulation steps
        self.optimizer = torch.optim.Adam
        self.default_coverage = False
        self.clip = 10.0
        self.lm_lr = 1e-4
        self.summ_lr = 1e-5
        self.weight_decay = 1e-5
        self.tf_ratio = 0.85
        self.lm_epochs = 300
        self.summarizer_epochs = 0
        self.save_epoch = 1  # Number of epochs after which start saving versions of the model
        
        if self.acc_size is None:
            self.acc_size = 0
 