import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, emb_dim]
        outputs, (hidden, cell) = self.lstm(embedded)

        # Fusionner les directions (forward + backward)
        hidden = torch.cat((hidden[0], hidden[1]), dim=1).unsqueeze(0)  # [1, batch_size, 2*hidden_dim]
        cell = torch.cat((cell[0], cell[1]), dim=1).unsqueeze(0)  # [1, batch_size, 2*hidden_dim]

        return outputs, hidden, cell  # outputs: [batch_size, seq_len, 2*hidden_dim]