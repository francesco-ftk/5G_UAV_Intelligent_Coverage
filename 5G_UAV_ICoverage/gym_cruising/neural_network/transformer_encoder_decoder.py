import torch.nn as nn


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, embed_dim=16):
        super(TransformerEncoderDecoder, self).__init__()

        # TRANSFORMER ENCODER-DECODER
        self.embedding = nn.Linear(2, embed_dim)  # Embedding per il Transformer encoder e decoder
        self.transformer_enocder_decoder = nn.Transformer(d_model=embed_dim)

    def forward(self, GU_positions, UAV_positions):
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
