import torch
import torch.nn as nn
import math
from torch import Tensor


# TODO add positional embeddings and Masks?

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class PositionalEncoding1(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=
        torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-
                                                                    math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class CustomTransformerEncoderDecoder(nn.Module):
    def __init__(self, embed_dim=16, num_heads=4, num_layers=2):
        super(CustomTransformerEncoderDecoder, self).__init__()
        self.embedding1 = nn.Linear(2, embed_dim)  # Embedding per il primo Transformer
        self.embedding2 = nn.Linear(2, embed_dim)  # Embedding per il secondo Transformer
        self.positional_encoding = PositionalEncoding(embed_dim)

        self.first_transformer = nn.Transformer(
            d_model=embed_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=0
        )

        self.second_transformer = nn.Transformer(
            d_model=embed_dim, nhead=num_heads, num_encoder_layers=0, num_decoder_layers=num_layers
        )

        # TODO Per ora inutile
        self.output_layer = nn.Linear(embed_dim, embed_dim)  # Strato di output se necessario

    def forward(self, x1, x2):
        # x1 shape: (n, 2)
        # x2 shape: (3, 2)

        # Embedding delle sequenze di input
        x1 = self.embedding1(x1)  # shape: (n, embed_dim)
        x2 = self.embedding2(x2)  # shape: (3, embed_dim)

        x3 = self.positional_encoding(x1)  # shape: (n, embed_dim)

        # Aggiungi dimensione batch
        x1 = x1.unsqueeze(1)  # shape: (n, 1, embed_dim)
        x2 = x2.unsqueeze(1)  # shape: (3, 1, embed_dim)

        # Passa la prima sequenza attraverso il primo Transformer (solo encoder)
        memory = self.first_transformer.encoder(x1)  # shape: (n, 1, embed_dim)

        # Passa la seconda sequenza attraverso il secondo Transformer (solo decoder) con il contesto
        output = self.second_transformer.decoder(x2, memory)  # shape: (3, 1, embed_dim)

        # Rimuovi la dimensione batch
        output = output.squeeze(1)  # shape: (3, embed_dim)

        # Applica uno strato di output se necessario
        # output = self.output_layer(output)  # shape: (3, embed_dim)

        return output
