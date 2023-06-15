import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, emb_dim, dec_dim):
        super().__init__()
        self.attn = nn.Linear(emb_dim + dec_dim, dec_dim)
        self.v = nn.Linear(dec_dim, 1, bias=False)
    
    def forward(self, hidden, seq_embs, mask):
        # hidden: [1, batch_size, dec_dim]
        # seq_embs: [seq_len, batch_size, emb_dim]
        # mask: [batch_size, seq_len]
        
        batch_size = seq_embs.shape[1]
        seq_len = seq_embs.shape[0]
        
        # repeat decoder hidden state seq_len times
        hidden = hidden.repeat(seq_len, 1, 1).permute(1, 0, 2) # [batch_size, seq_len, dec_dim]
        enc_outputs_cat = torch.cat((hidden, seq_embs.permute(1, 0, 2)), dim=2)  # [batch_size, dec_dim + emb_dim]
        energy = torch.tanh(self.attn(enc_outputs_cat)) # [batch_size, seq_len, dec_dim]
        
        attention = self.v(energy).squeeze(2)    # [batch_size, seq_len]
        attention = attention.masked_fill(mask==0, -1e10)
        attn_dist = F.softmax(attention, dim=1)  # [batch_size, seq_len]
        
        return attn_dist

# Source pyGAT: https://github.com/Diego999/pyGAT/
class GraphAttentionLayer(nn.Module):
    
    def __init__(self, input_dim, output_dim, dropout=0.5, alpha=0.2, concat=True, device="cpu"):
        super().__init__()
        self.dropout = dropout
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(input_dim, output_dim))).to(device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*output_dim, 1))).to(device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        # W: [hid_dim * num_layers, hid_dim * num_layers]
        # h: [seq_len, hid_dim * num_layers]
        Wh = torch.matmul(h, self.W) # [seq_len, hid_dim * num_layers]
        attention = self._prepare_attentional_mechanism_input(Wh)  # [seq_len, seq_len]
        attention = F.softmax(attention, dim=1) # [seq_len, seq_len]
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh) # [seq_len, hid_dim * num_layers]
        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        # a: [output_dim * 2, 1]
        Wh1 = torch.matmul(Wh, self.a[:self.output_dim, :])  # [seq_len, 1]
        Wh2 = torch.matmul(Wh, self.a[self.output_dim:, :])  # [seq_len, 1]
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e) 

class GAT(nn.Module):
    def __init__(self, device, input_dim, output_dim, hidden_dim=200, dropout=0.5, alpha=0.2, num_heads=2):
        super().__init__()
        self.device = device
        self.dropout = dropout
        self.num_heads = num_heads
        
        self.layers = [GraphAttentionLayer(input_dim, output_dim, dropout=dropout, alpha=alpha, concat=True, device=device) for _ in range(num_heads)]

    def forward(self, x):
        # x: [seq_len, hid_dim * num_layers]
        h_prime = F.dropout(x, self.dropout, training=self.training)
        for att in self.layers:
            h_prime = att(h_prime) # [seq_len, hid_dim * num_layers]
        return h_prime
