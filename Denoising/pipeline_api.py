import os
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from model import SimpleUNet

def run_mri_denoising_pipeline(input_path_run1, input_path_run2=None, output_path='denoised_output.nii.gz'):
    """
    Diese Funktion ist die zentrale Schnittstelle für das GitHub-Repository.
    Sie nimmt beliebige MRT-Eingangsdaten, skaliert sie automatisch auf 128x128x93,
    trainiert das Noise2Noise-Modell selbstüberwacht und exportiert das saubere Volumen.
    
    Parameters:
    - input_path_run1 (str): Pfad zur verrauschten NIfTI-Datei (Run 01)
    - input_path_run2 (str): Optional. Pfad zu Run 02 (für Noise2Noise). 
                             Falls None, wird Run 01 für ein Noise2Void-artiges Training kopiert.
    - output_path (str): Pfad, unter dem das fertige 128x128x93 Volumen gespeichert wird.
    
    Returns:
    - str: Pfad zur erfolgreich erstellten Ausgabedatei.
    """
    print("\n=========================================================================")
    print("🚀 STARTE MODULARE MRI-DENOISING-PIPELINE (Standardgröße: 128x128x93)")
    print("=========================================================================")

    # Falls die Gruppenmitglieder kein zweites Run-Volumen haben, spiegeln wir das erste
    if input_path_run2 is None:
        print("💡 Hinweis: Kein Run 02 übergeben. Nutze Single-Image Self-Supervision.")
        input_path_run2 = input_path_run1

    if not os.path.exists(input_path_run1) or not os.path.exists(input_path_run2):
        raise FileNotFoundError("Einer der angegebenen Eingangs-Pfade wurde nicht gefunden!")

    # --- Phase 1: Daten laden und anpassen ---
    print("\n[Phase 1/4] Lade Bilddaten und passe Dimensionen an...")
    img1_obj = nib.load(input_path_run1)
    img2_obj = nib.load(input_path_run2)
    
    # Selektives Laden des ersten 3D-Volumens (verhindert den 10.2 GB RAM Absturz)
    if len(img1_obj.shape) == 4:
        data1 = np.asanyarray(img1_obj.dataobj[:, :, :, 0])
        data2 = np.asanyarray(img2_obj.dataobj[:, :, :, 0])
    else:
        data1 = img1_obj.get_fdata()
        data2 = img2_obj.get_fdata()
        
    print(f"-> Eingangsgröße der Rohdaten: {data1.shape}")

    # Erzwinge strikte Standarddimensionen für die Gruppe
    target_shape = (128, 128, 93)
    zoom_factors = [t / o for t, o in zip(target_shape, data1.shape)]
    
    volume1_resized = zoom(data1, zoom_factors, order=1)
    volume2_resized = zoom(data2, zoom_factors, order=1)
    height, width, depth = volume1_resized.shape
    print(f"-> Ziel-Dimensionen erfolgreich erzwungen: {height}x{width}x{depth}")

    # --- Phase 2: KI-Training ---
    print("\n[Phase 2/4] Starte stabiles KI-Training (Anker- & Zufallsschichten)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.L1Loss()

    def norm_func(x, p_val=None): 
        if p_val is None: p_val = np.percentile(x, 99)
        if p_val == 0: p_val = 1
        return np.clip(x, 0, p_val) / p_val, p_val

    model.train()
    for epoch in range(1001):
        optimizer.zero_grad()
        
        if epoch % 2 == 0:
            z_active = 65  # Fester anatomischer Anker
        else:
            z_active = np.random.randint(30, 80)
            
        raw_a = volume1_resized[:, :, z_active]
        raw_b = volume2_resized[:, :, z_active]

        n_a, _ = norm_func(raw_a)
        n_b, _ = norm_func(raw_b)
        
        input_tensor = torch.from_numpy(n_a).unsqueeze(0).unsqueeze(0).float().to(device)
        target_tensor = torch.from_numpy(n_b).unsqueeze(0).unsqueeze(0).float().to(device)

        output = model(input_tensor)
        
        if np.random.rand() > 0.5:
            output = torch.flip(output, [2])
            target_tensor = torch.flip(target_tensor, [2])
            
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            mode = "ANKER" if epoch % 2 == 0 else "ZUFALL"
            print(f"   Epoche {epoch:4d}/1000 | {mode} | Loss: {loss.item():.6f}")

    # --- Phase 3: Export ---
    print("\n[Phase 3/4] Generiere rauschfreies 3D-Volumen...")
    denoised_3d = np.zeros((height, width, depth), dtype=np.float32)
    
    anchor_data = volume1_resized[:, :, 65]
    _, global_p = norm_func(anchor_data)

    model.eval()
    with torch.no_grad():
        for z in range(depth):
            curr_slice = volume1_resized[:, :, z]
            s_norm, _ = norm_func(curr_slice, p_val=global_p)
            
            inp = torch.from_numpy(s_norm).unsqueeze(0).unsqueeze(0).float().to(device)
            out = model(inp)
            
            denoised_3d[:, :, z] = out.cpu().squeeze().numpy() * global_p

    # --- Phase 4: Glättung & Speichern ---
    print("\n[Phase 4/4] Wende Post-Processing-Glättung an und speichere...")
    denoised_3d = gaussian_filter(denoised_3d, sigma=0.5)

    new_affine = np.eye(4)
    final_img = nib.Nifti1Image(denoised_3d, new_affine)
    nib.save(final_img, output_path)
    
    print("=========================================================================")
    print(f"✅ FERTIG! Saubere Datei gespeichert unter: {output_path} ({height}x{width}x{depth})")
    print("=========================================================================\n")
    
    return output_path

# Ermöglicht es, die Datei auch weiterhin direkt via "python pipeline_api.py" zu testen
if __name__ == "__main__":
    p1 = 'data/raw/sub-01_ses-0p9mm_dir-AP_run-01_part-mag_dwi.nii.gz'
    p2 = 'data/raw/sub-01_ses-0p9mm_dir-AP_run-02_part-mag_dwi.nii.gz'
    run_mri_denoising_pipeline(p1, p2, output_path='sub-01_denoised_full.nii.gz')