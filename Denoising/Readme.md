# MRI fMRI/DWI Denoising Module (Inference Endpoint)

Dieses GitHub-Modul bietet eine vollautomatische, speicherschonende **Deep-Learning-Pipeline (Inferenz)** zur hocheffizienten Rauschunterdrückung in funktionellen und strukturellen MRT-Daten (fMRI/DWI).

Das Modul (`pipeline_api.py`) ist als reiner Produktions-Endpoint konzipiert. Es lädt ein vortrainiertes U-Net und wendet es auf den gesamten 4D-Scan an, wobei die **originale native Scanner-Auflösung und Geometrie (Affine/Header)** zu 100 % erhalten bleiben.

---

## 🚀 Kern-Features
1. **Reiner Inference-Endpoint:** Strikt vom Training getrennt. Das Modul lädt fertige Gewichte (`.pth`) und entrauscht die Daten verlässlich, ohne jemals in einen Trainings-Loop zu fallen oder Modellgewichte zu überschreiben.
2. **Native Auflösung & Geometrie:** Kein künstliches Resizing (Stauchen/Strecken). Das 2D CNN passt sich dynamisch an jede Eingangsgröße an. Die Ausgangsdatei übernimmt exakt die Dimensionen sowie die originalen Header- und Affine-Metadaten des Scanners.
3. **Full 4D-Processing & RAM-Protection:** Verarbeitet iterativ *jeden einzelnen Zeitpunkt (T)* des 4D-Scans. Durch schichtweises Laden der Volumes (`dataobj`) und die Deaktivierung der Gradientenberechnung (`torch.no_grad()`) bleibt der Arbeitsspeicher-Bedarf selbst bei riesigen Dateien minimal.
4. **Per-Volume Normalisierung:** Nutzt das 99. Perzentil zur dynamischen Kontrastanpassung jedes Volumens, um Helligkeitsschwankungen über die Zeitachse perfekt auszugleichen.

---

## 📦 Voraussetzungen (Requirements)

Stellt sicher, dass die benötigten Pakete in eurer Python-Umgebung installiert sind:
```bash
pip install nibabel torch numpy




from pipeline_api import denoise_run

# Startet die Inference-Pipeline auf den rohen Scanner-Daten:
output_file = denoise_run(
    input_path='data/raw/sub-01_ses-00_task-ArchiSocial_dir-ap_bold.nii.gz',
    output_path='sub-01_denoised_full.nii.gz',
    weights_path='mri_unet_robust.pth' # Die vorab trainierte Modell-Datei
)

print(f"Das bereinigte 4D-MRT wurde erfolgreich gespeichert unter: {output_file}")