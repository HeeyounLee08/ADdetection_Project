import os, glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class AdniLongitudinalDataset(Dataset):
    """
    Dataset for (subject,visit) samples with MRI, PET, APOE, ADAS13-as-token.
    """
    def __init__(
        self,
        adas_csv: str,
        mri_meta_csv: str,
        pet_meta_csv: str,
        mri_root: str,
        pet_root: str,
        apoe_map: dict,
        label_map: dict,
        split: str = "train",
        val_frac: float = 0.2,
        seed: int = 42,
    ):

        df = pd.read_csv(adas_csv)
        df = df.dropna(subset=["DIAGNOSIS","TOTAL13"]).reset_index(drop=True)
        df["adas_norm"] = MinMaxScaler((0,1)).fit_transform(df[["TOTAL13"]])


        meta_mri = pd.read_csv(mri_meta_csv)   # subject_id, image_id, mri_visit
        meta_pet = pd.read_csv(pet_meta_csv)   # subject_id, image_id, pet_visit

   
        def build_lookup(root, meta_df, visit_col, is_mri):
            lut = {}
            for path in glob.glob(os.path.join(root, "*")):
                bn = os.path.basename(path)
                parts = bn.split("_")

                if is_mri:
                    # "ADNI_011_S_0003_...": parts[1:4]
                    subj = "_".join(parts[1:4])
                else:
                    # "011_S_0003_Ixxxxx_128": parts[0:3]
                    subj = "_".join(parts[0:3])
                # image_id 
                try:
                    img_id = int(bn.split("_I")[1].split("_")[0])
                except:
                    continue
                # Finding visit in meta
                row = meta_df[
                    (meta_df["subject_id"] == subj) &
                    (meta_df["image_id"]   == img_id)
                ]
                if len(row)==1:
                    visit = row[visit_col].iloc[0]
                    lut[(subj, visit)] = path
            return lut

        self.mri_lookup = build_lookup(mri_root, meta_mri, "mri_visit", is_mri=True)
        self.pet_lookup = build_lookup(pet_root, meta_pet, "pet_visit", is_mri=False)

        df = df[df.apply(lambda r: (r["subject_id"], r["visit"]) in self.mri_lookup
                                 and (r["subject_id"], r["visit"]) in self.pet_lookup, axis=1)]
        df = df.reset_index(drop=True)

        train_df, val_df = train_test_split(
            df, test_size=val_frac, random_state=seed, stratify=df["DIAGNOSIS"]
        )
        self.df = train_df if split=="train" else val_df
        self.apoe_map  = apoe_map
        self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        subj, visit = row["subject_id"], row["visit"]

        # MRI tokens
        mri_folder = self.mri_lookup[(subj,visit)]
        mri_files  = sorted(glob.glob(os.path.join(mri_folder,"slice_*.npy")))
        mri_toks   = np.concatenate([np.load(f).flatten() for f in mri_files])

        # PET tokens
        pet_folder = self.pet_lookup[(subj,visit)]
        pet_files  = sorted(glob.glob(os.path.join(pet_folder,"slice_*.npy")))
        pet_toks   = np.concatenate([np.load(f).flatten() for f in pet_files])

        # Combine MRI+PET
        enc_tokens = np.concatenate([mri_toks, pet_toks])
        enc_mods   = np.array([0]*len(mri_toks) + [1]*len(pet_toks))

        # APOE token (modality=2)
        apoe_id    = self.apoe_map[row["GENOTYPE"]]
        enc_tokens = np.concatenate([enc_tokens, [apoe_id]])
        enc_mods   = np.concatenate([enc_mods,   [2]])

        # ADAS13 token (0–100) as modality=3
        adas_token = int(row["adas_norm"] * 100)
        enc_tokens = np.concatenate([enc_tokens, [adas_token]])
        enc_mods   = np.concatenate([enc_mods,    [3]])

        # Positions & pad mask
        N         = len(enc_tokens)
        positions = np.arange(N)
        pad_mask  = np.ones(N, dtype=bool)

        # Label
        label = self.label_map[str(row["DIAGNOSIS"])]

        return {
            "tokens":     torch.LongTensor(enc_tokens),
            "modalities": torch.LongTensor(enc_mods),
            "positions":  torch.LongTensor(positions),
            "pad_mask":   torch.BoolTensor(pad_mask),
            "label":      torch.LongTensor([label]),
        }
