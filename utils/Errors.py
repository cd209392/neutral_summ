import torch.nn as nn
import torch.nn.functional as F
import torch

def loss_estimation(targets, outputs, pad_idx):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    criterion = nn.NLLLoss(ignore_index=pad_idx)
    #outputs = torch.log(outputs)
    rec_loss = criterion(outputs, targets)
    return rec_loss