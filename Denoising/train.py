import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from model import SimpleUNet

def train_and_export():
    path1 = 'data/raw/sub-01_ses-0p9mm_dir-AP_run-01_part-mag_dwi.nii.gz'
    path2 = 'data/raw/sub-01_ses-0p9mm_dir-AP_run-02_part-mag_dwi.nii.gz'
    output_name = 'sub-01_denoised_full.nii.gz'

    print("--- Phase 1: Daten speicherschonend laden und auf 128x128x93 skalieren ---")
    img1_obj = nib.load(path1)
    img2_obj = nib.load(path2)
    
    # SPEICHER-RETTUNG: Lade NUR das erste 3D-Volumen (Index 0) direkt von der Festplatte,
    # anstatt die vollen 10.2 GB in den RAM zu knallen!
    print("Lese erstes 3D-Volumen der Rohdaten...")
    if len(img1_obj.shape) == 4:
        data1 = np.asanyarray(img1_obj.dataobj[:, :, :, 0])
        data2 = np.asanyarray(img2_obj.dataobj[:, :, :, 0])
    else:
        data1 = img1_obj.get_fdata()
        data2 = img2_obj.get_fdata()
    
    print(f"Originale Dimensionen des extrahierten Volumens: {data1.shape}")
    
    # Hier erzwingen wir deine Wunsch-Größe für Input & Output
    target_shape = (128, 128, 93)
    zoom_factors = [t / o for t, o in zip(target_shape, data1.shape)]
    
    print("Skaliere Daten im Zwischenspeicher auf 128 x 128 x 93...")
    volume1_resized = zoom(data1, zoom_factors, order=1)
    volume2_resized = zoom(data2, zoom_factors, order=1)
    
    height, width, depth = volume1_resized.shape
    print(f"Ziel-Dimensionen erreicht: {height}x{width}x{depth}")
    
    # Zufallsbereich für die Slices (angepasst an die neue Tiefe 93)
    z_min, z_max = 30, 80 

    model = SimpleUNet()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.L1Loss()

    def norm_func(x, p_val=None): 
        if p_val is None:
            p_val = np.percentile(x, 99)
        if p_val == 0: p_val = 1
        return np.clip(x, 0, p_val) / p_val, p_val

    print("\n--- Phase 2: Stabiles Training (Anker + Zufall) ---")
    for epoch in range(1001):
        optimizer.zero_grad()
        
        if epoch % 2 == 0:
            z_active = 65 
        else:
            z_active = np.random.randint(z_min, z_max)
            
        raw_a = volume1_resized[:, :, z_active]
        raw_b = volume2_resized[:, :, z_active]

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
        
        if epoch % 101 == 0:
            mode = "ANKER" if epoch % 2 == 0 else "ZUFALL"
            print(f"Epoche {epoch} | {mode} | Schicht {z_active} | Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), 'mri_unet_robust.pth')

    print("\n--- Phase 3: Export mit Globaler Normalisierung ---")
    denoised_3d = np.zeros((height, width, depth), dtype=np.float32)
    
    anchor_data = volume1_resized[:, :, 65]
    _, global_p = norm_func(anchor_data)

    model.eval()
    with torch.no_grad():
        for z in range(depth):
            curr_slice = volume1_resized[:, :, z]
            s_norm, _ = norm_func(curr_slice, p_val=global_p)
            
            inp = torch.from_numpy(s_norm).unsqueeze(0).unsqueeze(0).float()
            out = model(inp)
            
            denoised_3d[:, :, z] = out.squeeze().numpy() * global_p
            
            if z % 20 == 0:
                print(f"Export Schicht {z}/{depth} (Scale: {global_p:.1f})")

    print("\n--- Phase 4: Post-Processing (Final Smoothing) ---")
    denoised_3d = gaussian_filter(denoised_3d, sigma=0.5)

    new_affine = np.eye(4)
    final_img = nib.Nifti1Image(denoised_3d, new_affine)
    nib.save(final_img, output_name)
    print(f"\nERFOLG! Datei abgespeichert unter: {output_name} mit Größe {denoised_3d.shape}")

if __name__ == "__main__":
    train_and_export()