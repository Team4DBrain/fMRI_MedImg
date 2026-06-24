# MRI fMRI/DWI Denoising Module (All-in-One Pipeline)

Dieses GitHub-Modul bietet eine vollautomatische, speicherschonende **Noise2Noise Deep-Learning-Pipeline** zur hocheffizienten Rauschunterdrückung in funktionellen und strukturellen MRT-Daten (fMRI/DWI).

Das gesamte Modul ist in einer einzigen Datei (`pipeline_api.py`) gekapselt. Es akzeptiert Gehirn-Scans in jeglicher Auflösung, standardisiert sie vollautomatisch im Arbeitsspeicher auf eine feste Ziel-Dimension von **128 x 128 x 93 Voxeln**, trainiert ein integriertes U-Net und exportiert die bereinigten Daten fehlerfrei.

---

## 🚀 Kern-Features
1. **Zentrales All-in-One Modul:** Die gesamte Architektur (U-Net), Logik (Noise2Noise) und Vorverarbeitung liegen in einer einzigen Datei. Die Gruppe muss nichts konfigurieren oder verschieben.
2. **Automatische Standardisierung:** Egal welche Dimensionen die Rohdaten eurer Gruppenmitglieder haben (z. B. `228x228x132`), das Modul interpoliert sie vor dem Training exakt auf **128 x 128 x 93** für Input und Output.
3. **RAM-Protection (Lazy Loading):** Durch selektiven Zugriff auf das erste 3D-Volumen via `dataobj` wird verhindert, dass Python bei großen 4D-Dateien abstürzt (Speicherbedarf sinkt von ~10.2 GB auf wenige Megabyte).
4. **Anker-Training:** Das Modell wechselt im Trainings-Loop intelligent zwischen einer festen mittleren anatomischen Schicht (Schicht 65) und Zufallsschichten, um strukturelle Geometrien perfekt zu erhalten.

---

## 📦 Voraussetzungen (Requirements)

Stellt sicher, dass die benötigten Pakete in eurer Python-Umgebung (oder `.venv`) installiert sind:
```bash
pip install nibabel scipy torch numpy



from pipeline_api import run_mri_denoising_pipeline

# Startet das Laden, Resizing, Training und den sauberen Export mit einem Befehl:
output_file = run_mri_denoising_pipeline(
    input_path_run1='data/raw/sub-01_ses-0p9mm_dir-AP_run-01_part-mag_dwi.nii.gz',
    input_path_run2='data/raw/sub-01_ses-0p9mm_dir-AP_run-02_part-mag_dwi.nii.gz', # Optional
    output_path='sub-01_denoised_full.nii.gz'
)

print(f"Das bereinigte 128x128x93 Volumen wurde gespeichert unter: {output_file}")