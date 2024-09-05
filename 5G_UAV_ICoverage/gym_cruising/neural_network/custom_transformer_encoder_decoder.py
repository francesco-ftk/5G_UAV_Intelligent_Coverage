import torch
import torch.nn as nn
import math
from torch import Tensor

class CustomTransformerEncoderDecoder(nn.Module):
    def __init__(self, embed_dim=16, num_heads=4, num_layers=2):
        super(CustomTransformerEncoderDecoder, self).__init__()
        self.embedding1 = nn.Linear(2, embed_dim)  # Embedding per il primo Transformer
        self.embedding2 = nn.Linear(2, embed_dim)  # Embedding per il secondo Transformer

        self.first_transformer = nn.Transformer(
            d_model=embed_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=0
        )

        self.second_transformer = nn.Transformer(
            d_model=embed_dim, nhead=num_heads, num_encoder_layers=0, num_decoder_layers=num_layers
        )

    def forward(self, x1, x2):
        # x1 shape: (n, 2) GU
        # x2 shape: (3, 2) UAV

        # Embedding delle sequenze di input
        x1 = self.embedding1(x1)  # shape: (n, embed_dim)
        x2 = self.embedding2(x2)  # shape: (3, embed_dim)

        # Aggiungi dimensione batch
        x1 = x1.unsqueeze(1)  # shape: (n, 1, embed_dim)
        x2 = x2.unsqueeze(1)  # shape: (3, 1, embed_dim)

        # Passa la prima sequenza attraverso il primo Transformer (solo encoder)
        memory = self.first_transformer.encoder(x1)  # shape: (n, 1, embed_dim)

        # Passa la seconda sequenza attraverso il secondo Transformer (solo decoder) con il contesto
        output = self.second_transformer.decoder(x2, memory)  # shape: (3, 1, embed_dim)

        # Rimuovi la dimensione batch
        output = output.squeeze(1)  # shape: (3, embed_dim)

        return output
