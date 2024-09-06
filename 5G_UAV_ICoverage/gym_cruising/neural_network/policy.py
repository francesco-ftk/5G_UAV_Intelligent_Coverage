import torch
import torch.nn as nn
import math
import random
from torch import Tensor

EPS_START = 0.9  # the starting value of epsilon
EPS_END = 0.3  # the final value of epsilon
EPS_DECAY = 60000  # controls the rate of exponential decay of epsilon, higher means a slower decay


class PolicyNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, seq_len: int = 1, n_layers: int = 1, embed_dim=16):
        super(PolicyNet, self).__init__()

        # TRANSFORMER ENCODER-DECODER
        self.embedding = nn.Linear(2, embed_dim)  # Embedding per il Transformer encoder e decoder
        self.transformer_enocder_decoder = nn.Transformer(d_model=embed_dim)

        # LSTM
        self.input_size = input_size
        self.latent_size = hidden_size
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.lstm = nn.LSTM(self.input_size, self.latent_size, self.n_layers, batch_first=True)
        self.fl1 = nn.Linear(self.latent_size, output_size)

    def forward(self, GU_positions, UAV_positions, UAV_number, time_steps_done):
        # GU_positions shape: (n, 2)
        # UAV_positions shape: (m, 2)

        # Embedding delle sequenze di input
        GU_positions = self.embedding(GU_positions)  # shape: (n, embed_dim)
        UAV_positions = self.embedding(UAV_positions)  # shape: (m, embed_dim)

        # Aggiungi dimensione batch
        source = GU_positions.unsqueeze(1)  # shape: (n, 1, embed_dim)
        target = UAV_positions.unsqueeze(1)  # shape: (m, 1, embed_dim)

        # RAPPRESENTAZIONE DELLO STATO PER OGNI UAV
        tokens = self.transformer_enocder_decoder(source, target)  # shape: (m, 1, embed_dim)

        # Rimuovi la dimensione batch
        tokens = tokens.squeeze(1)  # shape: (m, embed_dim)

        return tokens
