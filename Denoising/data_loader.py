import nibabel as nib
import torch
import numpy as np

def load_real_data():
    # Load the 4D volumes
    img1 = nib.load('data/raw/sub-01_ses-0p9mm_dir-AP_run-01_part-mag_dwi.nii.gz').get_fdata()
    img2 = nib.load('data/raw/sub-01_ses-0p9mm_dir-AP_run-02_part-mag_dwi.nii.gz').get_fdata()

    # MRI check: These files are (X, Y, Z, Gradient). 
    # Let's take the middle slice (e.g., 30) of the first gradient volume (0).
    slice_a = img1[:, :, 30, 0]
    slice_b = img2[:, :, 30, 0]

    # Normalize to 0-1 range
    def norm(x): return (x - x.min()) / (x.max() - x.min())
    
    tensor_a = torch.from_numpy(norm(slice_a)).unsqueeze(0).unsqueeze(0).float()
    tensor_b = torch.from_numpy(norm(slice_b)).unsqueeze(0).unsqueeze(0).float()
    
    return tensor_a, tensor_b