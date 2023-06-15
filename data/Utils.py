#!/usr/bin/env python
# coding: utf-8


import os
import pickle
import torch
import random
import json
import pandas as pd
from tqdm import tqdm
import spacy


nlp = spacy.load("en_core_web_sm")


def collate_tokens(values, pad_idx, left_pad=False, pad_to_length=None):
    # Simplified version of `collate_tokens` from fairseq.data.data_utils
    """Convert a list of 1d tensors into a padded 2d tensor."""
    values = list(map(torch.LongTensor, values))
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def tokenize(text):
    return [str(word) for word in nlp(str(text))]


def rev_max_len(list_revs):
    """Getting max length of all reviews to padd all sentences to same size in fields"""
    max_len = 0
    for review in list_revs:
        tok_list = tokenize(review)
        rev_len = len(tok_list)
        if rev_len > max_len:
            max_len = rev_len

    return max_len