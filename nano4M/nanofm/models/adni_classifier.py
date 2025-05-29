import torch
import torch.nn as nn
import torch.nn.functional as F
from nanofm.modeling.transformer_layers import TransformerTrunk, LayerNorm

class AdniClassifier(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        depth: int = 8,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        num_classes: int = 3,
        max_seq_len: int = 514
    ):
        super().__init__()

        orig_vocab_size = 64107
        PAD_ID = orig_vocab_size
        self.PAD_ID = PAD_ID
        # ── token / modality / position embeddings ──────────────────────────
        self.enc_tok_emb = nn.Embedding(num_embeddings=orig_vocab_size+1, embedding_dim=dim, padding_idx=PAD_ID )
        self.enc_mod_emb = nn.Embedding(num_embeddings=4,   embedding_dim=dim)  # MRI,PET,APOE,ADAS
        self.pos_emb     = nn.Parameter(torch.randn(max_seq_len, dim))         # big enough
        self.PAD_ID = PAD_ID

        self.register_buffer(             # ← NEW
            "mod_offsets",
            torch.tensor([0, 0, 64_000, 64_006], dtype=torch.long),
            persistent=False,
        )

        # ── Transformer encoder trunk ───────────────────────────────────────
        self.encoder = TransformerTrunk(
            dim=dim,
            depth=depth,
            head_dim=head_dim,
            mlp_ratio=mlp_ratio,
        )

        # ── classification head ─────────────────────────────────────────────
        self.classifier = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_classes),
        )

    # --------------------------------------------------------------------- #
    # `run_training.py` expects the model to return  (loss, metrics_dict)
    # --------------------------------------------------------------------- #
    def forward(self, batch):
        enc_tokens = batch["enc_tokens"]   # (B, L)
        enc_mods   = batch["enc_mods"]     # (B, L)
        enc_pos    = batch["enc_pos"]      # (B, L)
        enc_mask   = batch["enc_mask"]     # (B, L)  True = real token

        ids = enc_tokens + self.mod_offsets[enc_mods]
        # 1) embeddings + encoder
        # x = (
        #     self.enc_tok_emb(enc_tokens)
        #     + self.enc_mod_emb(enc_mods)
        #     + self.pos_emb[:, enc_pos, :]
        # )

        pos_emb = self.pos_emb[enc_pos]
        x = self.enc_tok_emb(enc_tokens) \
          + self.enc_mod_emb(enc_mods)  \
          + pos_emb
        # x = self.encoder(x, mask=enc_mask)        # (B, L, D)
        seq_len = enc_tokens.size(1)
        real_len = (enc_tokens != self.PAD_ID).sum(dim=1)        # (B,)
        arange = torch.arange(seq_len, device=enc_tokens.device).unsqueeze(0)  # (1, L)
        pad_mask = arange < real_len.unsqueeze(1)           # True = real, False = pad
        attn_mask = pad_mask.unsqueeze(1) & enc_mask.unsqueeze(2)
        x = self.encoder(x, mask=attn_mask)

        # 2) mean-pool → head
        feat   = x.mean(dim=1)                    # (B, D)
        logits = self.classifier(feat)            # (B, C)

        # 3) loss & simple metric
        labels  = batch["label"].view(-1)         # (B,)
        loss    = F.cross_entropy(logits, labels)
        acc     = (logits.argmax(dim=-1) == labels).float().mean()
        metrics = {"acc": acc}

        return loss, metrics
