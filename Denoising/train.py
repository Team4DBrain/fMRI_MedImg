import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import SimpleUNet

def train_and_export():
    path1 = 'data/raw/sub-01_ses-0p9mm_dir-AP_run-01_part-mag_dwi.nii.gz'
    path2 = 'data/raw/sub-01_ses-0p9mm_dir-AP_run-02_part-mag_dwi.nii.gz'
    output_name = 'sub-01_denoised_full.nii.gz'

    print("--- Phase 1: Stabiles Training (Anker + Zufall) ---")
    img1_obj = nib.load(path1)
    img2_obj = nib.load(path2)
    
    z_min, z_max = 40, 90 

    model = SimpleUNet()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.L1Loss()

    def norm_func(x, p_val=None): 
        if p_val is None:
            p_val = np.percentile(x, 99)
        if p_val == 0: p_val = 1
        return np.clip(x, 0, p_val) / p_val, p_val

    for epoch in range(1001):
        optimizer.zero_grad()
        
        # Wechsel zwischen Anker und Zufall
        if epoch % 2 == 0:
            z_active = 65 
        else:
            z_active = np.random.randint(z_min, z_max)
            
        raw_a = np.asanyarray(img1_obj.dataobj[:, :, z_active, 0])
        raw_b = np.asanyarray(img2_obj.dataobj[:, :, z_active, 0])

        n_a, _ = norm_func(raw_a)
        n_b, _ = norm_func(raw_b)
        
        input_tensor = torch.from_numpy(n_a).unsqueeze(0).unsqueeze(0).float()
        target_tensor = torch.from_numpy(n_b).unsqueeze(0).unsqueeze(0).float()

        output = model(input_tensor)
        
        if np.random.rand() > 0.5:
            output = torch.flip(output, [2])
            target_tensor = torch.flip(target_tensor, [2])
            
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()
        
        # Korrigierter Print für abwechselnde Anzeige
        if epoch % 101 == 0:
            mode = "ANKER" if epoch % 2 == 0 else "ZUFALL"
            print(f"Epoche {epoch} | {mode} | Schicht {z_active} | Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), 'mri_unet_robust.pth')

    print("\n--- Phase 2: Export mit Globaler Normalisierung ---")
    height, width, depth, _ = img1_obj.shape
    denoised_3d = np.zeros((height, width, depth), dtype=np.float32)
    
    anchor_data = np.asanyarray(img1_obj.dataobj[:, :, 65, 0])
    _, global_p = norm_func(anchor_data)

    model.eval()
    with torch.no_grad():
        for z in range(depth):
            curr_slice = np.asanyarray(img1_obj.dataobj[:, :, z, 0])
            s_norm, _ = norm_func(curr_slice, p_val=global_p)
            
            inp = torch.from_numpy(s_norm).unsqueeze(0).unsqueeze(0).float()
            out = model(inp)
            
            denoised_3d[:, :, z] = out.squeeze().numpy() * global_p
            
            if z % 20 == 0:
                print(f"Export Schicht {z}/{depth} (Scale: {global_p:.1f})")

    print("\n--- Phase 3: Post-Processing (Final Smoothing) ---")
    from scipy.ndimage import gaussian_filter
    
    # Sigma 0.5 glättet die Schichtsprünge perfekt weg
    denoised_3d = gaussian_filter(denoised_3d, sigma=0.5)

    # NIfTI erstellen und speichern
    final_img = nib.Nifti1Image(denoised_3d, img1_obj.affine, img1_obj.header)
    nib.save(final_img, output_name)
    print(f"\nERFOLG! Datei überschrieben: {output_name}")

if __name__ == "__main__":
    train_and_export()