import os
import glob
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.ndimage import zoom
from model import SimpleUNet

def train_on_real_pairs():
    print("\n=========================================================================")
    print("🧠 STARTE ECHTES NOISE2NOISE PAAR-TRAINING (AP <-> PA & Run1 <-> Run2)")
    print("=========================================================================")

    # --- Kugelsicherer Pfad & Speicherort ---
    data_dir = r"C:\Users\sugie\mri_project\data\raw"
    weights_path = 'mri_unet_robust.pth'

    print(f"-> Suche Daten im festen Ordner: {data_dir}")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Der Ordner {data_dir} existiert nicht. Bitte Pfad prüfen!")

    # --- Phase 1: Perfekte Paare finden ---
    all_files = glob.glob(os.path.join(data_dir, '*.nii*'))
    print(f"-> HABE {len(all_files)} MRT-DATEIEN IN DIESEM ORDNER GEFUNDEN.")

    if len(all_files) == 0:
        print("\n⚠️ STOPP: Der Ordner ist komplett leer! Die Bilder liegen wahrscheinlich woanders.")
        return
        
    training_pairs = []
    
    for file1 in all_files:
        file2 = None
        f1_lower = file1.lower()
        
        if 'dir-ap' in f1_lower:
            target_name = os.path.basename(file1).lower().replace('dir-ap', 'dir-pa')
            for possible_match in all_files:
                if os.path.basename(possible_match).lower() == target_name:
                    file2 = possible_match
                    break
                    
        elif 'run-01' in f1_lower:
            target_name = os.path.basename(file1).lower().replace('run-01', 'run-02')
            for possible_match in all_files:
                if os.path.basename(possible_match).lower() == target_name:
                    file2 = possible_match
                    break

        if file2:
            pair = (file1, file2)
            if pair not in training_pairs and (file2, file1) not in training_pairs:
                training_pairs.append(pair)

    print(f"-> {len(training_pairs)} perfekte MRT-Paare für das Training gefunden!")

    # --- Phase 2: Modell und Hardware vorbereiten ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleUNet().to(device)

    if os.path.exists(weights_path):
        print(f"\n-> 📂 Lade existierende Gewichte: '{weights_path}'")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("\n-> 🆕 Keine Basis gefunden. Lerne von Null an.")

    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.L1Loss()
    model.train()

    def norm_func(x): 
        p_val = np.percentile(x, 99)
        if p_val == 0: p_val = 1
        return np.clip(x, 0, p_val) / p_val, p_val

    target_shape = (128, 128, 93)

    # =================================================================================
    # TEIL 1: DER GARANTIERTE 100% DURCHLAUF (Sequenziell)
    # =================================================================================
    print("\n🚀 TEIL 1 STARTE: Sequenzieller Durchlauf (Jedes Volume wird exakt 1x gelernt)")
    
    for pair_idx, (path_a, path_b) in enumerate(training_pairs):
        name_a = os.path.basename(path_a)
        print(f"\n   -> Lese gesamtes Paar [{pair_idx+1}/{len(training_pairs)}]: {name_a.split('_dir')[0]}...")
        
        try:
            img_a = nib.load(path_a)
            img_b = nib.load(path_b)
            
            num_volumes = min(img_a.shape[3] if len(img_a.shape) == 4 else 1, 
                              img_b.shape[3] if len(img_b.shape) == 4 else 1)
            
            # Zähler für die Epochen
            epochen_pro_paar = 0
            
            # Gehe rigoros von Volume 0 bis Ende durch
            for v in range(num_volumes):
                vol_a = np.asanyarray(img_a.dataobj[:, :, :, v]) if len(img_a.shape) == 4 else img_a.get_fdata()
                vol_b = np.asanyarray(img_b.dataobj[:, :, :, v]) if len(img_b.shape) == 4 else img_b.get_fdata()
                
                zoom_factors_a = [t / o for t, o in zip(target_shape, vol_a.shape)]
                zoom_factors_b = [t / o for t, o in zip(target_shape, vol_b.shape)]
                
                res_a = zoom(vol_a, zoom_factors_a, order=1)
                res_b = zoom(vol_b, zoom_factors_b, order=1)
                
                for epoch in range(4):
                    optimizer.zero_grad()
                    z_active = np.random.randint(20, 80) 
                    
                    n_a, _ = norm_func(res_a[:, :, z_active])
                    n_b, _ = norm_func(res_b[:, :, z_active])
                    
                    if np.random.rand() > 0.5:
                        in_t = torch.from_numpy(n_a).unsqueeze(0).unsqueeze(0).float().to(device)
                        tar_t = torch.from_numpy(n_b).unsqueeze(0).unsqueeze(0).float().to(device)
                    else:
                        in_t = torch.from_numpy(n_b).unsqueeze(0).unsqueeze(0).float().to(device)
                        tar_t = torch.from_numpy(n_a).unsqueeze(0).unsqueeze(0).float().to(device)
                    
                    if np.random.rand() > 0.5:
                        in_t = torch.flip(in_t, [2])
                        tar_t = torch.flip(tar_t, [2])
                        
                    out = model(in_t)
                    loss = criterion(out, tar_t)
                    loss.backward()
                    optimizer.step()
                    epochen_pro_paar += 1
            
            # Sichern nach jedem komplett gelesenen Paar + Saubere Ausgabe
            torch.save(model.state_dict(), weights_path)
            print(f"   💾 PAAR {pair_idx+1} ABGESCHLOSSEN: {epochen_pro_paar} Epochen trainiert. (Letzter Loss: {loss.item():.6f})")
            
        except Exception as e:
            print(f"      ❌ Fehler beim Verarbeiten von {name_a}: {e}")
            continue

    print("\n✅ TEIL 1 ERFOLGREICH BEENDET: Das Netz hat nun das gesamte Dataset gesehen!")


    # =================================================================================
    # TEIL 2: DER ENDLOSE ZUFALLS-LOOP (Zur Perfektionierung)
    # =================================================================================
    print("\n🚀 TEIL 2 STARTE: Endloser Zufalls-Loop (Abbruch jederzeit mit STRG+C)")
    global_loop = 1
    
    while True:
        print(f"\n🌀 ZUFALLS-RUNDE {global_loop}")
        np.random.shuffle(training_pairs) 
        
        for pair_idx, (path_a, path_b) in enumerate(training_pairs):
            name_a = os.path.basename(path_a)
            epochen_pro_paar = 0
            
            try:
                img_a = nib.load(path_a)
                img_b = nib.load(path_b)
                
                num_volumes = min(img_a.shape[3] if len(img_a.shape) == 4 else 1, 
                                  img_b.shape[3] if len(img_b.shape) == 4 else 1)
                
                volumes_to_sample = min(5, num_volumes)
                random_vols = np.random.choice(num_volumes, volumes_to_sample, replace=False)
                
                for v in random_vols:
                    vol_a = np.asanyarray(img_a.dataobj[:, :, :, v]) if len(img_a.shape) == 4 else img_a.get_fdata()
                    vol_b = np.asanyarray(img_b.dataobj[:, :, :, v]) if len(img_b.shape) == 4 else img_b.get_fdata()
                    
                    zoom_factors_a = [t / o for t, o in zip(target_shape, vol_a.shape)]
                    zoom_factors_b = [t / o for t, o in zip(target_shape, vol_b.shape)]
                    
                    res_a = zoom(vol_a, zoom_factors_a, order=1)
                    res_b = zoom(vol_b, zoom_factors_b, order=1)
                    
                    for epoch in range(4):
                        optimizer.zero_grad()
                        z_active = np.random.randint(20, 80)
                        
                        n_a, _ = norm_func(res_a[:, :, z_active])
                        n_b, _ = norm_func(res_b[:, :, z_active])
                        
                        if np.random.rand() > 0.5:
                            in_t = torch.from_numpy(n_a).unsqueeze(0).unsqueeze(0).float().to(device)
                            tar_t = torch.from_numpy(n_b).unsqueeze(0).unsqueeze(0).float().to(device)
                        else:
                            in_t = torch.from_numpy(n_b).unsqueeze(0).unsqueeze(0).float().to(device)
                            tar_t = torch.from_numpy(n_a).unsqueeze(0).unsqueeze(0).float().to(device)
                        
                        if np.random.rand() > 0.5:
                            in_t = torch.flip(in_t, [2])
                            tar_t = torch.flip(tar_t, [2])
                            
                        out = model(in_t)
                        loss = criterion(out, tar_t)
                        loss.backward()
                        optimizer.step()
                        epochen_pro_paar += 1
                        
                print(f"      -> Paar {pair_idx+1} aktualisiert: {epochen_pro_paar} Epochen trainiert (Loss: {loss.item():.6f})")
                        
            except Exception as e:
                pass 
        
        torch.save(model.state_dict(), weights_path)
        print(f"   💾 RUNDE {global_loop} BEENDET & GESICHERT.")
        global_loop += 1

if __name__ == "__main__":
    train_on_real_pairs()
