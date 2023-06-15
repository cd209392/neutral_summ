#!/usr/bin/env python
# coding: utf-8

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import GloVe
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")


def freeze(module):
    for param in module.parameters():
        param.requires_grad = False

def logits_to_prob(logits, method, tau=1.0, eps=1e-10, gumbel_hard=False):
    """
    Args:
        logits: [batch_size, vocab_size]
        method: 'gumbel', 'softmax'
        gumbel_hard: boolean
        topk: int (used for beam search)
    Returns: [batch_size, vocab_size]
    """
    if tau == 0.0:
        raise ValueError(
            'Temperature should not be 0. If you want greedy decoding, pass "greedy" to prob_to_vocab_id()')
    if method == 'gumbel':
        prob = F.gumbel_softmax(logits, tau=tau, eps=eps, hard=gumbel_hard)
    elif method == 'softmax':
        prob = F.log_softmax(logits / tau, dim=1)
    return prob

#####################################
# Masking
#####################################

#For masking padded parts of sentences in attention layer
def indiv_mask(src, src_pad_idx):
    mask = (src != src_pad_idx).permute(1, 0)
    return mask

def mask_range(src, pos, src_pad_idx):
    mask = indiv_mask(src, src_pad_idx)  # [batch_size, seq_len]
    mask[:, pos:-1] = False
    return mask

#####################################
# Weight Initialization
#####################################

def xavier_weights_init(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param.data)
        else:
            nn.init.constant_(param.data, 0)
            
def normal_weights_init(m):
    mean = 0
    std = 0.01
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=mean, std=std)

#####################################
# Loading Embedding representation
#####################################


def load_gloveembeddings(vocab, Glove_name='6B', Glove_dim=100):
    emb_vecs = []
    glove_vecs = GloVe(name=Glove_name, dim=Glove_dim, unk_init=torch.Tensor.normal_)

    for w in vocab._id_to_word:
        w_emb = glove_vecs[w]
        emb_vecs.append(w_emb)

    emb_vecs = list(map(lambda x: x.squeeze(), emb_vecs))
    emb_vecs = torch.stack(emb_vecs)

    return emb_vecs


#####################################
# Projection Mechanism
#####################################

def project_vector(V, N, device):
    # V: [batch_size, dec_dim]
    # N: [batch_size, dec_dim]
    W = []
    batch_size = V.size(0)
    for b_idx in range(batch_size):
        v = V[b_idx] # [dec_dim]
        n = N[b_idx] # [dec_dim]
        # Compute norm of vector n
        norm = torch.norm(n)
        # Trim vectors to same size to do the dot product
        vn = torch.dot(v, torch.t(n))
        # Multiply (v.n / n) * (n / n)
        w = (vn / norm) * (n / norm) # [dec_dim]
        W.append(w.unsqueeze(0))  
    W = torch.vstack(W).to(device) # [batch_size, dec_dim] 
    return W


#####################################
# Cosine Similarity
#####################################
def cosine_similarity(v, n):
    # v, n: [dec_dim]
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    score = cos(v, n)
    return score


#####################################
# Postprocess
#####################################
def postprocess(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True):
    if skip_special_tokens:
        tokens = [t for t in tokens if not is_special(t)]

    out_string = ' '.join(tokens)

    if clean_up_tokenization_spaces:
         out_string = clean_up_tokenization(out_string)

    return out_string


def is_special(token):
    res = re.search("\<[a-z]+\>", token)
    if res is None:
        return False
    return token == res.group()


def clean_up_tokenization(out_string):
    """
    Reference : transformers.tokenization_utils_base
    Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.
    Args:
        out_string (:obj:`str`): The text to clean up.
    Returns:
        :obj:`str`: The cleaned-up string.
    """
    out_string = (
        out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
    )
    return out_string
    