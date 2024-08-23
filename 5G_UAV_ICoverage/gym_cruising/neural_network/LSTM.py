import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len: int = 1, n_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.latent_size = hidden_size
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.lstm = nn.LSTM(self.input_size, self.latent_size, self.n_layers, batch_first=True)
        self.fl1 = nn.Linear(self.latent_size, output_size)

    def forward(self, x, batch_size, hidden_state):
        cell_state = torch.zeros(self.n_layers, batch_size, self.latent_size)
        hidden = (hidden_state, cell_state)
        x = x.reshape([batch_size, self.seq_len, self.input_size])
        y, hidden = self.lstm(x, hidden)
        return self.fl1(hidden[0]), hidden
