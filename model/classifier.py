import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, hidden):
        # hidden: [batch_size, dec_dim]
        preds = torch.sigmoid(self.fc2(F.relu(self.fc1(hidden)))).squeeze(1)
        return preds  # [batch_size]