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

    def forward(self, current_state, batch_size, hidden_state, cell_state):
        hidden = (hidden_state, cell_state)
        current_state = current_state.reshape([self.seq_len, batch_size, self.input_size])
        y, hidden = self.lstm(current_state, hidden)
        return self.fl1(hidden[0]), hidden
