#!/usr/bin/env python
"""
prep_adni.py
============
End-to-end preprocessing that converts raw ADNI T1-MRI ZIP archives
into 128 × 128 × 128 NumPy tensors suitable for deep-learning.

Pipeline
--------
1.  Unzip every `Cohort_*.zip` (skipped if already extracted).
2.  N4 bias-field correction (SimpleITK, CPU):
      • shrink factor 2       → 4× fewer voxels
      • iteration schedule    → (20, 20, 20, 10)
3.  Brain extraction with **HD-BET** (GPU, “folder mode”).
4.  Min–max normalisation **inside** the brain mask; background = 0.
5.  Isotropic resample to (128, 128, 128) with linear interpolation.
6.  Save as `.npy` (float32) to OUT_DIR.
7.  Remove temporary N4 / BET files to save disk space.

Cluster features
----------------
* Designed for **SLURM job arrays**.  
  Files are distributed by `index % SLURM_ARRAY_TASK_COUNT == task_id`.
* Sets `MKL_THREADING_LAYER=GNU` once to avoid the NumPy × ITK OpenMP clash.
* HD-BET runs on the single GPU visible in the allocation; N4 uses as many
  CPU threads as `OMP_NUM_THREADS`.

Requirements (inside your conda env)
------------------------------------
conda install  -y  pytorch torchvision pytorch-cuda=11.8  -c pytorch -c nvidia  
pip   install     nibabel SimpleITK hd-bet scipy pandas
"""

# ─────────────────────────────────────────────────────────────
import os, zipfile, glob, subprocess
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")        # OpenMP clash fix

import nibabel as nib
import numpy as np
from  scipy.ndimage  import zoom
import SimpleITK as sitk
import gzip
import shutil


RAW_DIR = "/scratch/izar/helee/adni_data"          # original ZIPs / .nii
OUT_DIR = "/scratch/izar/helee/adni_preproc_128"   # final .npy files
TARGET  = (128, 128, 128)                          # final voxel grid
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────── utilities ──────────────────────

def compress_nii(nii_path: str) -> str:
    gz_path = nii_path + ".gz"
    with open(nii_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return gz_path


def unzip_archives() -> None:
    """Extract every ZIP exactly once."""
    for z in glob.glob(f"{RAW_DIR}/*.zip"):
        dst = z[:-4]                                # …/file.zip → …/file/
        if not os.path.isdir(dst):
            print("[UNZIP]", os.path.basename(z))
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dst)

def run_n4(src: str, dst: str,
           shrink: int = 2,
           iters: tuple[int, int, int, int] = (20, 20, 20, 10)) -> None:
    """
    N4 with manual shrink:
      1) Down-sample   (shrink)
      2) Run N4 on small image
      3) Up-sample bias-field    ↑
    """
    img = sitk.ReadImage(src)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)

    # 1) shrink
    if shrink > 1:
        img_small  = sitk.Shrink(img , [shrink]*img.GetDimension())
        mask_small = sitk.Shrink(mask, [shrink]*mask.GetDimension())
    else:
        img_small, mask_small = img, mask

    # 2) N4 on the small image
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations(iters)
    corrected_small = n4.Execute(img_small, mask_small)

    # 3) apply bias field to FULL resolution image
    log_bias_field  = n4.GetLogBiasFieldAsImage(img_small)
    log_bias_full   = sitk.Resample(log_bias_field, img, sitk.Transform(),
                                    sitk.sitkLinear, 0.0, log_bias_field.GetPixelID())
    corrected_full  = img / sitk.Exp(log_bias_full)
    sitk.WriteImage(corrected_full, dst)

def norm(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min–max normalise inside *mask*; keep background at 0."""
    brain = vol[mask > 0]
    out   = (vol - brain.min()) / (brain.max() - brain.min())
    out[mask == 0] = 0.0
    return out

def resize(vol: np.ndarray) -> np.ndarray:
    """Linear-interpolate *vol* to TARGET shape."""
    factors = [t / s for t, s in zip(TARGET, vol.shape)]
    return zoom(vol, factors, order=1)

# ───────────────────── per-file pipeline ────────────────────
def process_one_nii(nii: str) -> None:
    """Convert a single NIfTI file to 128³ `.npy`."""
    base   = os.path.basename(nii).split(".nii")[0]
    npyout = f"{OUT_DIR}/{base}_128.npy"
    if os.path.exists(npyout):
        return                                           # already processed

    print("base   = ", base)
    # 1) N4 bias correction  →  *.nii  (uncompressed for speed)
    n4 = nii.replace(".nii.gz", "_n4.nii").replace(".nii", "_n4.nii")
    print("n4 = nii.replace : ", n4)
    run_n4(nii, n4)
    gz_n4 = compress_nii(n4)
    print(gz_n4)

    # 2) HD-BET (folder mode, GPU)
    
    out_prefix = os.path.splitext(gz_n4)[0]       # strips ONLY .gz → …_n4.nii
    out_prefix = os.path.splitext(out_prefix)[0]  # strips .nii    → …_n4

    print(" out_prefix = ", out_prefix)
    subprocess.run(
        ["hd-bet", "-i", gz_n4,
         "-device", "0"],
         check=True,
    )

    bet  = out_prefix + "_bet.nii.gz"             # …_n4_bet.nii.gz
    mask = out_prefix + "_bet_mask.nii.gz"        # …_n4_bet_mask.nii.gz
    print("bet: ", bet)
    print("mask:", mask)

    # 3) load → normalise → resize → save
    vol128 = resize(norm(nib.load(bet).get_fdata(),
                     nib.load(mask).get_fdata())).astype(np.float32)
    print("vol129 resize done")
    np.save(npyout, vol128)
    print("[SAVE]", os.path.basename(npyout), vol128.shape)

    # 4) clean-up temp files
    for f in (n4, bet, mask):
        if os.path.exists(f):
            os.remove(f)

# ─────────────────────────── main ───────────────────────────
if __name__ == "__main__":
    unzip_archives()

    nii_list = sorted(glob.glob(f"{RAW_DIR}/**/*.nii*", recursive=True))
    print("Total NIfTI files found:", len(nii_list))

    # SLURM array support  (defaults to single-task when launched manually)
    task_id  = int(os.getenv("SLURM_ARRAY_TASK_ID",  "0"))
    task_cnt = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))

    for idx, n in enumerate(nii_list):
        if idx % task_cnt == task_id:
            process_one_nii(n)
