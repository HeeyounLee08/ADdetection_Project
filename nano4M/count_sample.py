import os, glob
import pandas as pd
from sklearn.model_selection import train_test_split
from nanofm.data.adni.adni_longitudinal_dataset import AdniLongitudinalDataset 


adas_csv     = "/work/com-304/adni/adas13.csv"
df = pd.read_csv(adas_csv)
df = df.dropna(subset=["DIAGNOSIS","TOTAL13"]).reset_index(drop=True)


train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["DIAGNOSIS"]
)
print(f"Total num of samples: {len(df)}")
print(f"Train samples: {len(train_df)}")
print(f"Val   samples: {len(val_df)}")


ds_train = AdniLongitudinalDataset(
    adas_csv=adas_csv,
    mri_meta_csv="/work/com-304/adni/meta_MRI_images.csv",
    pet_meta_csv="/work/com-304/adni/meta_PET_images.csv",
    mri_root="/work/com-304/adni/cosmos_mri_tokens",
    pet_root="/work/com-304/adni/pet_cosmos_tokens",
    apoe_map={"2/2":0,"2/3":1,"2/4":2,"3/3":3,"3/4":4,"4/4":5},
    label_map={"1.0":0,"2.0":1,"3.0":2},
    split="train",
)
ds_val = AdniLongitudinalDataset(
    adas_csv=adas_csv,
    mri_meta_csv="/work/com-304/adni/meta_MRI_images.csv",
    pet_meta_csv="/work/com-304/adni/meta_PET_images.csv",
    mri_root="/work/com-304/adni/cosmos_mri_tokens",
    pet_root="/work/com-304/adni/pet_cosmos_tokens",
    apoe_map={"2/2":0,"2/3":1,"2/4":2,"3/3":3,"3/4":4,"4/4":5},
    label_map={"1.0":0,"2.0":1,"3.0":2},
    split="val",
)
print(f"Dataset(train) sample count: {len(ds_train)}")
print(f"Dataset(val) sample count: {len(ds_val)}")