import torch
from torch.nn.utils.rnn import pad_sequence

def adni_collate_fn(batch):
    toks  = [b["tokens"]     for b in batch]
    mods  = [b["modalities"] for b in batch]
    pos   = [b["positions"]  for b in batch]
    mask  = [b["pad_mask"]   for b in batch]
    labels= torch.cat([b["label"] for b in batch], dim=0)

    return {
      "enc_tokens": pad_sequence(toks,  batch_first=True, padding_value=0),
      "enc_mods":   pad_sequence(mods,  batch_first=True, padding_value=0),
      "enc_pos":    pad_sequence(pos,   batch_first=True, padding_value=0),
      "enc_mask":   pad_sequence(mask,  batch_first=True, padding_value=False),
      "label":      labels,
    }

def pad_collate(batch):
    from . import adni_collate_fn
    return adni_collate_fn(batch)
