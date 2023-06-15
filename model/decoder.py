import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, emb_dim, dropout=0.5):
        super().__init__()
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, embedded, hidden, context):
        # embedded: [1, batch_size, emb_dim]
        # hidden = [1, batch_size, hid_dim]
        # context = [1, batch_size, hid_dim]
        emb_con = torch.cat((embedded, context), dim = 2) # [1, batch_size, emb_dim + hid_dim]
        output, hidden = self.rnn(emb_con, hidden)
        # output = [1, batch_size, hid_dim]
        # hidden = [1, batch_size, hid_dim]
        
        #assert (output == hidden).all()
        
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim = 1) # [batch_size, emb_dim + hid_dim * 2]
        prediction = self.fc_out(output) # [batch_size, output_dim]
        
        return prediction, hidden.squeeze(0)