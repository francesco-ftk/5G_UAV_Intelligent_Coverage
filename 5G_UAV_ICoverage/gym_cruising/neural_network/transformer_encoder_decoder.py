import torch.nn as nn


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, embed_dim=16):
        super(TransformerEncoderDecoder, self).__init__()

        # TRANSFORMER ENCODER-DECODER
        self.embedding_encoder = nn.Linear(2, embed_dim)  # Embedding per il Transformer encoder
        self.embedding_decoder = nn.Linear(4, embed_dim)  # Embedding per il Transformer decoder
        self.transformer_enocder_decoder = nn.Transformer(d_model=embed_dim)

    def forward(self, GU_positions, UAV_info):
        # GU_positions shape: (n, 2)
        # UAV_info shape: (m, 4)

        # Embedding delle sequenze di input
        GU_positions = self.embedding_encoder(GU_positions)  # shape: (n, embed_dim)
        UAV_info = self.embedding_decoder(UAV_info)  # shape: (m, embed_dim)

        # Aggiungi dimensione batch
        source = GU_positions.unsqueeze(1)  # shape: (n, 1, embed_dim)
        target = UAV_info.unsqueeze(1)  # shape: (m, 1, embed_dim)

        # RAPPRESENTAZIONE DELLO STATO PER OGNI UAV
        tokens = self.transformer_enocder_decoder(source, target)  # shape: (m, 1, embed_dim)

        # Rimuovi la dimensione batch
        tokens = tokens.squeeze(1)  # shape: (m, embed_dim)

        return tokens  # TODO tutta la gestione batch va fatta fuori forse
