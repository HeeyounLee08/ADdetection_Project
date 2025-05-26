#!/usr/bin/env python
# scripts/test_adni_dataset.py

import os
from torch.utils.data import DataLoader
from nanofm.data.adni.adni_longitudinal_dataset import AdniLongitudinalDataset
from nanofm.data.adni.collate_fn import adni_collate_fn
def main():
    # ———————————— 변경할 부분 ————————————
    adas_csv     = "/work/com-304/adni/adas13.csv"
    mri_meta_csv = "/work/com-304/adni/meta_MRI_images.csv"
    pet_meta_csv = "/work/com-304/adni/meta_PET_images.csv"
    mri_root     = "/work/com-304/adni/cosmos_mri_tokens"
    pet_root     = "/work/com-304/adni/pet_cosmos_tokens"
    apoe_map     = {"2/2":0, "2/3":1, "2/4":2, "3/3":3, "3/4":4, "4/4":5}
    label_map    = {"1.0":0, "2.0":1, "3.0":2}
    split        = "train"   # 또는 "val"
    batch_size   = 2

    # ——————————————————————————————————————–

    # 1) Dataset 인스턴스화
    ds = AdniLongitudinalDataset(
        adas_csv, mri_meta_csv, pet_meta_csv,
        mri_root, pet_root,
        apoe_map, label_map,
        split=split
    )
    print(f">>> Loaded ADNI dataset ({split} split), {len(ds)} samples\n")

    # 2) 첫 샘플 하나 보기
    sample = ds[0]
    print("=== Sample[0] keys & shapes ===")
    for k,v in sample.items():
        v_str = f"{tuple(v.shape)}" if hasattr(v, "shape") else str(v)
        # 만약 토큰 벡터이라면 앞 10개만 출력
        if k=="tokens":
            v_show = v.numpy()[:10].tolist()
            v_str += f"  (first 10 IDs: {v_show} …)"
        print(f"  {k:10s}: {v_str}")
    print()

    # 3) DataLoader 로 배치 단위도 확인
    loader = DataLoader(ds,
                        batch_size=batch_size,
                        collate_fn=adni_collate_fn,
                        shuffle=False)
    batch = next(iter(loader))
    print("=== Batch shapes ===")
    for k,v in batch.items():
        print(f"  {k:10s}: {tuple(v.shape)}")
    print()

    # 4) 간단한 값 분포 확인 (토큰 ID, modality ID, label 분포)
    import numpy as np
    all_labels = [ds[i]["label"].item() for i in range(min(50, len(ds)))]
    print("Sample labels (first 50):", all_labels)
    print("Unique labels:", np.unique(all_labels).tolist())

if __name__ == "__main__":
    main()

