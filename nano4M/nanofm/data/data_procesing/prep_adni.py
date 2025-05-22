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
from typing import Sequence


RAW_DIR = "/scratch/izar/helee/adni_data/ADNI"          # original ZIPs / .nii
OUT_DIR = "/scratch/izar/helee/adni_preproc_128"   # final .npy files
TARGET  = (128, 128, 128)                          # final voxel grid
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────── utilities ──────────────────────

def compress_nii(nii_path: str) -> str:
    gz_path = nii_path + ".gz"
    with open(nii_path, 'rb') as f_in, gzip.open(gz_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    return gz_path






def run_n4(src: str, dst: str,
           spline_distance: int = 50,
           iters: Sequence[int] = (50, 40, 30, 20),
           conv: float = 1e-7) -> None:
    """
    N4 bias-field correction at full resolution.

    Parameters
    ----------
    src : str
        Path to input NIfTI (.nii or .nii.gz).
    dst : str
        Output filename (same extension as src).
    spline_distance : int, optional
        Control-point spacing (mm). Larger ⇒ faster, smoother field.
    iters : 4-tuple[int], optional
        Number of iterations per level.
    conv : float
        Convergence threshold.
    """
    img  = sitk.ReadImage(src, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)

    # --- tell N4 to use a coarse B-spline grid instead of shrinking voxels ---
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations(iters)
    n4.SetConvergenceThreshold(conv)

    # approximate grid spacing in voxels
    spacing_mm   = img.GetSpacing()
    img_size     = img.GetSize()
    grid_nodes   = [
        max(2, int(round(sz * sp / spline_distance))) for sz, sp in zip(img_size, spacing_mm)
    ]
    n4.SetSplineOrder(3)
    n4.SetBiasFieldFullWidthAtHalfMaximum(0.15)
    n4.SetNumberOfControlPoints(grid_nodes)

    # --- run N4 ---
    corrected = n4.Execute(img, mask)

    sitk.WriteImage(corrected, dst)

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

    # 1) N4 bias correction  →  *.nii  (uncompressed for speed)
    n4 = nii.replace(".nii.gz", "_n4.nii").replace(".nii", "_n4.nii")
    try:
        run_n4(nii, n4)                        
    except RuntimeError as e:
        if "control points" in str(e):
            print("[N4] control-point error → retry with spline_distance=30")
            run_n4(nii, n4, spline_distance=30) 
        else:
            raise  
    gz_n4 = compress_nii(n4)

    # 2) HD-BET (folder mode, GPU)
    
    out_prefix = os.path.splitext(gz_n4)[0]       # strips ONLY .gz → …_n4.nii
    out_prefix = os.path.splitext(out_prefix)[0]  # strips .nii    → …_n4

    subprocess.run(
         ["hd-bet", "-i",  gz_n4, "-device", "0"],
         check=True,
    )




    bet  = out_prefix + "_bet.nii.gz"             # …_n4_bet.nii.gz
    mask = out_prefix + "_bet_mask.nii.gz"        # …_n4_bet_mask.nii.gz

    # 3) load → normalise → resize → save
    vol128 = resize(norm(nib.load(bet).get_fdata(),
                     nib.load(mask).get_fdata())).astype(np.float32)
    np.save(npyout, vol128)
    print("[SAVE]", os.path.basename(npyout), vol128.shape)

    # 4) clean-up temp files
    for f in (n4, bet, mask):
        if os.path.exists(f):
            os.remove(f)

# ─────────────────────────── main ───────────────────────────
if __name__ == "__main__":

    nii_list = sorted(glob.glob(f"{RAW_DIR}/**/*.nii*", recursive=True))
    print("Total NIfTI files found:", len(nii_list))

    # SLURM array support  (defaults to single-task when launched manually)
    task_id  = int(os.getenv("SLURM_ARRAY_TASK_ID",  "0"))
    task_cnt = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))

    for idx, n in enumerate(nii_list):
        if idx % task_cnt == task_id:
            process_one_nii(n)
