#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim

from utils.model_utils import xavier_weights_init

ROOT_DIR = "./"

path = {
    "root": ROOT_DIR,
    "data": f"{ROOT_DIR}data/",
    "train": f"{ROOT_DIR}data/processed/train.min.csv",
    "valid": f"{ROOT_DIR}data/processed/valid.min.csv",
    "test": f"{ROOT_DIR}data/processed/test.csv"
}

data = {
    "vocab_size": 5000,  #50 000
    "min_freq": 1,
    "workers": 1,
    "remove_stopwords": True,
    "remove_punctuation": True,
    "pin_memory": False,
    "workers": 1,
    "truncate": False,
    "max_num_reviews": 15
}

model = {
    "weights_init": xavier_weights_init,
    "enable_rec": True,
    "enable_cls": False,
    "enable_grl": False,
    "enable_proj": False,
    "enable_summ": False,
    "input__num_docs": 2,
    "input__cat_docs": False,
    "emb__emb_dim": 100,
    "emb__use_pretrained": True,
    "emb__Glove_name": "6B",
    "enc__hidden_dim": 302,
    "enc__drop": 0.2,
    "enc__num_layers": 2,
    "enc__bidir": True,
    "enc__use_gat": False,
    "dec__hidden_dim": 303,
    "dec__drop": 0.2,
    "dec__num_layers": 1,
    "dec__bidir": False,
    "dec__use_H_tilt": False,
    "gat__hidden_dim": 104,
    "gat__num_heads": 5,
    "gat__alpha": 0.2,
    "gat__drop": 0.5,
    "cls__hidden_dim": 200,
    "cls__num_classes": 2,
    "grl__gamma": 10.0
}

proc = {
    "optimizer" : torch.optim.Adam,
    "default_coverage" : False,
    "clip" : 10.0,
    "lr" : 1e-3,  #1e-3,                         #True : 0.0008 for Yelp and 0.0001 on Amazo
    "weight_decay" : 5e-5
}

error = {
    "criterion": nn.NLLLoss
}

follow = {
    "writer" : True,
    "Name" : "__rec_only__2docs",
    "write_rec_loss" : True,
    "write_total_loss" : True,
    "write_ROUGE" : True
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

assert (model["enable_rec"] and model["enable_summ"]) or (not model["enable_rec"] and not model["enable_summ"]) or model["enable_rec"]