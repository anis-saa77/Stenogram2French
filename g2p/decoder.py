import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, encoder_hidden_dim, decoder_hidden_dim, attention, num_layers=1):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim + encoder_hidden_dim * 2, decoder_hidden_dim*2, num_layers, batch_first=True)
        self.fc_out = nn.Linear(decoder_hidden_dim*2 + encoder_hidden_dim * 2 + emb_dim, output_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch_size] current target token
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.embedding(input)  # [batch_size, 1, emb_dim]

        attn_weights = self.attention(hidden[-1], encoder_outputs)  # [batch_size, src_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, src_len]

        context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, 2*encoder_hidden_dim]

        lstm_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, emb_dim + 2*encoder_hidden_dim]
        # print("hidden shape : ", hidden.shape)
        # print("lstm_input shape : ",lstm_input.shape)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        output = torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1)
        prediction = self.fc_out(output)  # [batch_size, output_dim]

        return prediction, hidden, cell, attn_weights.squeeze(1)