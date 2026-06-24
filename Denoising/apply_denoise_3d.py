import nibabel as nib
import torch
import numpy as np
from model import SimpleUNet

def denoise_3d_volume():
    # 1. Load the original noisy file
    path = 'data/raw/sub-01_ses-0p9mm_dir-AP_run-01_part-mag_dwi.nii.gz'
    img_obj = nib.load(path)
    data = img_obj.get_fdata() # This is (228, 228, 132, 199)
    
    # We will denoise the first volume (index 0) for now
    volume = data[:, :, :, 0] 
    height, width, depth = volume.shape
    denoised_volume = np.zeros_like(volume)

    # 2. Load your trained model weights (assuming you saved them)
    # For now, we use the model currently in memory if running in the same script, 
    # but let's assume we're applying the logic slice-by-slice:
    model = SimpleUNet() 
    # (In a real scenario, we'd load the .pth file here)
    model.eval() 

    print(f"Denoising {depth} slices...")
    with torch.no_grad():
        for z in range(depth):
            slice_2d = volume[:, :, z]
            
            # Normalize slice
            p_high = np.percentile(slice_2d, 99)
            if p_high == 0: p_high = 1 # Avoid division by zero
            norm_slice = np.clip(slice_2d, 0, p_high) / p_high
            
            # Convert to tensor
            input_t = torch.from_numpy(norm_slice).unsqueeze(0).unsqueeze(0).float()
            
            # Run AI
            output_t = model(input_t)
            
            # Convert back to numpy and rescale to original brightness
            denoised_slice = output_t.squeeze().numpy() * p_high
            denoised_volume[:, :, z] = denoised_slice
            
            if z % 20 == 0:
                print(f"Progress: {z}/{depth} slices done")

    # 3. Save as a new NIfTI file
    new_img = nib.Nifti1Image(denoised_volume, img_obj.affine, img_obj.header)
    nib.save(new_img, 'sub-01_denoised_full.nii.gz')
    print("Success! Saved to sub-01_denoised_full.nii.gz")

if __name__ == "__main__":
    denoise_3d_volume()