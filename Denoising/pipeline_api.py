import os
import numpy as np
import nibabel as nib
import torch

# Keep the SimpleUNet class import exactly as it is
from model import SimpleUNet 

def denoise_run(input_path, output_path, weights_path='mri_unet_robust.pth'):
    """Load a PRETRAINED denoiser and denoise an ENTIRE 4D run. No training here."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Load the model + PRETRAINED weights. Never train. Error if weights missing.
    model = SimpleUNet().to(device)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Pretrained weights '{weights_path}' not found. Train ONCE offline and "
            "save them first. This endpoint must never train."
        )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval() # Locks the model for inference only

    # 2) Open the run; KEEP its real affine + header (geometry). Do NOT resize.
    img = nib.load(input_path)
    affine = img.affine
    header = img.header.copy()
    shape = img.shape
    
    if len(shape) == 4:
        X, Y, Z, T = shape
    else:
        X, Y, Z = shape
        T = 1

    def norm_func(x, p_val=None):
        if p_val is None:
            p_val = np.percentile(x, 99)
        if p_val == 0:
            p_val = 1
        return np.clip(x, 0, p_val) / p_val, p_val

    # 3) Denoise EVERY timepoint (read one volume at a time to save memory),
    #    slice by slice, and stack into a 4D output.
    out = np.zeros((X, Y, Z, T), dtype=np.float32)
    
    with torch.no_grad(): # Explicitly disables gradient calculations to save memory and prevent training
        for t in range(T):
            if len(shape) == 4:
                vol = np.asanyarray(img.dataobj[..., t], dtype=np.float32)
            else:
                vol = np.asanyarray(img.dataobj, dtype=np.float32)
                
            _, p = norm_func(vol[:, :, Z // 2])   # one normalization scale per volume
            
            for z in range(Z):
                s_norm, _ = norm_func(vol[:, :, z], p_val=p)
                inp = torch.from_numpy(s_norm).unsqueeze(0).unsqueeze(0).float().to(device)
                out[:, :, z, t] = model(inp).cpu().squeeze().numpy() * p
                
            print(f"denoised volume {t + 1}/{T}")

    # 4) Save with the ORIGINAL affine + header (NOT np.eye(4)).
    nib.save(nib.Nifti1Image(out, affine, header), output_path)
    print(f"saved 4D denoised run -> {output_path}  shape={out.shape}")
    
    return output_path

if __name__ == "__main__":
    # Test execution block
    # Using your actual file paths for immediate testing
    test_input = r'C:\Users\sugie\mri_project\data\raw\sub-01_ses-00_task-ArchiSocial_dir-ap_bold.nii.gz'
    test_output = 'out.nii.gz'
    
    print("\n🚀 Starting strictly inference-only pipeline...")
    denoise_run(input_path=test_input, output_path=test_output)
