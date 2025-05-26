import torch.nn as nn
import torch
from modeling.transformer_layers import TransformerTrunk, LayerNorm

class AdniClassifier(nn.Module):
    def __init__(self, dim=512, depth=8, head_dim=64, mlp_ratio=4., num_classes=3):
        super().__init__()
        # Embeddings
        self.enc_tok_emb = nn.Embedding(  # vocab_size should cover max token+1, e.g. 101
            num_embeddings=101, embedding_dim=dim
        )
        self.enc_mod_emb = nn.Embedding(4, dim)   # MRI(0),PET(1),APOE(2),ADAS13(3)
        self.pos_emb     = nn.Parameter(torch.randn(1, 64000, dim))  

        # Encoder trunk
        self.encoder = TransformerTrunk(dim=dim, depth=depth, head_dim=head_dim, mlp_ratio=mlp_ratio)

        # Classification head
        self.classifier = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, num_classes)
        )

    def forward(self, enc_tokens, enc_mods, enc_pos, enc_mask):
        # 1) Embedding
        x = self.enc_tok_emb(enc_tokens) + self.enc_mod_emb(enc_mods) + self.pos_emb[:, enc_pos, :]
        # 2) Encode
        x = self.encoder(x, mask=enc_mask)
        # 3) Mean-pool
        feat = x.mean(dim=1)
        # 4) Classify
        return self.classifier(feat)
