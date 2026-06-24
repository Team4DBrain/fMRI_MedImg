import os
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.ndimage import zoom, gaussian_filter

# =========================================================================
# 1. KI-MODELL ARCHITEKTUR (U-Net)
# =========================================================================
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch), 
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        return self.final(d1)


# =========================================================================
# 2. CENTRAL PIPELINE FUNCTION (Transfer Learning / Fine-Tuning)
# =========================================================================
def run_mri_denoising_pipeline(input_path_run1, input_path_run2=None, output_path='sub-01_denoised_full.nii.gz', weights_path='mri_unet_robust.pth'):
    """
    Zentrale All-in-One Schnittstelle.
    Nutzt vorhandene Gewichte als Hintergrund-Basis (Transfer Learning) und trainiert
    sie direkt auf den neuen Eingangsdaten weiter, um das Modell weiter zu optimieren.
    """
    print("\n=========================================================================")
    print("🚀 PIPELINE-MODUL: AUTOMATISCHES MRI-DENOISING (Ziel: 128x128x93)")
    print("=========================================================================")

    if not os.path.exists(input_path_run1):
        raise FileNotFoundError(f"Eingangsdatei {input_path_run1} nicht gefunden!")

    # --- Phase 1: Speicherschonendes Laden & Resizing ---
    print("\n[Phase 1/4] Lade Daten stückweise & erzwinge Standardgröße...")
    img1_obj = nib.load(input_path_run1)
    
    if len(img1_obj.shape) == 4:
        data1 = np.asanyarray(img1_obj.dataobj[:, :, :, 0])
    else:
        data1 = img1_obj.get_fdata()
        
    print(f"-> Erkannte Eingangsgröße der Rohdaten: {data1.shape}")

    # Interpolation auf die festen Wunschdimensionen
    target_shape = (128, 128, 93)
    zoom_factors = [t / o for t, o in zip(target_shape, data1.shape)]
    
    print("-> Passe Dimensionen mathematisch an...")
    volume1_resized = zoom(data1, zoom_factors, order=1)
    height, width, depth = volume1_resized.shape
    print(f"-> Ziel-Dimensionen stabil eingestellt: {height}x{width}x{depth}")

    # Für Noise2Noise das zweite Volumen vorbereiten
    if input_path_run2 is None:
        input_path_run2 = input_path_run1
    img2_obj = nib.load(input_path_run2)
    data2 = np.asanyarray(img2_obj.dataobj[:, :, :, 0]) if len(img2_obj.shape) == 4 else img2_obj.get_fdata()
    volume2_resized = zoom(data2, zoom_factors, order=1)

    # Modell auf Hardware laden
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleUNet().to(device)

    # =========================================================================
    # INTENT: HINTERGRUND-GEWICHTE LADEN (TRANSFER LEARNING)
    # =========================================================================
    if os.path.exists(weights_path):
        print(f"\n[🔄 Hintergrund-Wissen] Lade existierende Gewichte '{weights_path}' als Basis...")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("-> Basis-Wissen geladen. Starte gezieltes Weiter-Training (Fine-Tuning)! ⚡")
    else:
        print(f"\n[⚠️ Start bei Null] Keine Gewichte unter '{weights_path}' gefunden. Trainiere komplett neu.")

    # --- Phase 2: Weiter-Training (Fine-Tuning) ---
    print("\n[Phase 2/4] Trainiere U-Net auf neuen Daten weiter...")
    
    # Kleineres Learning Rate (lr=0.0002 statt 0.0005), damit die geladenen Gewichte 
    # nicht zerstört werden, sondern sich ganz feinfühlig anpassen!
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.L1Loss()

    def norm_func(x, p_val=None): 
        if p_val is None: p_val = np.percentile(x, 99)
        if p_val == 0: p_val = 1
        return np.clip(x, 0, p_val) / p_val, p_val

    model.train()
    for epoch in range(1001):
        optimizer.zero_grad()
        
        # Anker-Wechsel für maximale Stabilität
        z_active = 65 if epoch % 2 == 0 else np.random.randint(30, 80)
            
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

    # Speichert die noch besseren, weiterentwickelten Gewichte wieder ab
    torch.save(model.state_dict(), weights_path)
    print(f"-> Aktualisierte Gewichte erfolgreich unter '{weights_path}' gesichert.")

    # --- Phase 3: Inferenz & Rekonstruktion ---
    print("\n[Phase 3/4] Generiere rauschfreies Gehirnvolumen...")
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

    # --- Phase 4: Filterung & Finale Ausgabe ---
    print("\n[Phase 4/4] Glätte Schichtübergänge und exportiere NIfTI...")
    denoised_3d = gaussian_filter(denoised_3d, sigma=0.5)

    new_affine = np.eye(4)
    final_img = nib.Nifti1Image(denoised_3d, new_affine)
    nib.save(final_img, output_path)
    
    print("=========================================================================")
    print(f"✅ ERFOLG! Sauberes Volumen gespeichert: {output_path} ({height}x{width}x{depth})")
    print("=========================================================================\n")
    
    return output_path

if __name__ == "__main__":
    p1 = 'data/raw/sub-01_ses-0p9mm_dir-AP_run-01_part-mag_dwi.nii.gz'
    p2 = 'data/raw/sub-01_ses-0p9mm_dir-AP_run-02_part-mag_dwi.nii.gz'
    run_mri_denoising_pipeline(p1, p2)
