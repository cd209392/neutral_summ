import torch.nn as nn
import torch.nn.functional as F
import torch

# ajouter max_oov_len
def loss_estimation(src_ext, final_dists, attn_dists, senti_outputs, senti_labels, pad_idx):
    # For sentiment detection: Cross Entropy Loss
    cross_criterion = nn.CrossEntropyLoss()
    cross_loss = cross_criterion(senti_outputs, senti_labels)
    
    # Negative log likelihood loss for reconstruction
    trg = src_ext[:, 1:]
    
    output = final_dists[:, :, 1:]
    output = torch.log(output)
    criterion = nn.NLLLoss
    criterion = criterion(ignore_index=pad_idx)
    rec_loss = criterion(output, trg)

    return rec_loss, cross_loss