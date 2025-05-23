import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attn = nn.Linear(encoder_hidden_dim * 2 + decoder_hidden_dim * 2, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, decoder_hidden_dim * 2]
        # encoder_outputs: [batch_size, seq_len, 2*encoder_hidden_dim]

        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, decoder_hidden_dim*2]
        attn = self.attn(torch.cat((hidden, encoder_outputs), dim=2)) # concat : [batch_size, seq_len, decoder_hidden_dim]
        energy = torch.tanh(attn)  # [batch_size, seq_len, decoder_hidden_dim]
        attention = self.v(energy).squeeze(2)  # [batch_size, seq_len]
        return F.softmax(attention, dim=1)