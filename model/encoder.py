import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, emb_dim, gat_attn=None, num_layers=1, dropout=0.5):
        super().__init__()
        self.hid_dim = hid_dim
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers=num_layers, bidirectional=True)
        self.fc = nn.Linear(hid_dim*2, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.gat_attn = gat_attn
        
    def forward(self, embedded, device):
        # embedded: [seq_len, batch_size, emb_dim]
        outputs, hidden = self.rnn(embedded)
        # outputs: [seq_len, batch_size, hid_dim * 2]
        # hidden: [2 * num_layers, batch_size, hid_dim]
        if self.gat_attn is None:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) # [batch_size, hid_dim * 2]
            hidden = torch.tanh(self.fc(hidden)) # [batch_size, hid_dim]
        else:
            # Extract the last hidden state of each direction of the BiRNN and use them as new hidden states
            # https://discuss.pytorch.org/t/clarification-regarding-the-return-of-nn-gru/47363/2
            batch_size = outputs.size(1)
            seq_len = outputs.size(0)
            hiddens = torch.empty_like(outputs).copy_(outputs).to(device)  # [seq_len, batch_size, hid_dim * 2]
            hiddens = hiddens.permute(1, 0, 2) # [batch_size, seq_len, hid_dim * 2]
            # Apply GAT Layer on the hidden state for each word
            hiddens = torch.vstack([self.gat_attn(hw).unsqueeze(0) for hw in hiddens]).permute(1, 0, 2) # [batch_size, seq_len, hid_dim * 2]
            # Separate forward and backward hidden states
            hiddens = hiddens.reshape((seq_len, batch_size, 2, -1)) # [seq_len, batch_size, 2, hid_dim]
            # Use the last hidden state in both directions
            hidden = torch.concat([hiddens[-1, :, -2, :], hiddens[0, :, -1, :]], dim=1) # [batch_size, hid_dim * 2]
            hidden = torch.tanh(self.fc(hidden)) # [batch_size, hid_dim]
        
        return hidden