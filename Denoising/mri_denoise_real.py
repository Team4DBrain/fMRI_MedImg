import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import SimpleUNet # Using the U-Net you created earlier

def train_real_n2n():
    print("Loading Real MRI Data (this might take a moment)...")
    # 1. Path to your downloaded files
    path1 = 'data/raw/sub-01_ses-0p9mm_dir-AP_run-01_part-mag_dwi.nii.gz'
    path2 = 'data/raw/sub-01_ses-0p9mm_dir-AP_run-02_part-mag_dwi.nii.gz'

    # 2. Load NIfTI and get data
    # Note: .get_fdata() loads the whole thing into RAM. 
    # If your PC slows down, we can use 'mmap' instead.
    img1 = nib.load(path1).get_fdata()
    img2 = nib.load(path2).get_fdata()

    # 3. Pick a slice with good brain structure (e.g., middle of the head)
    # The shape is likely (Height, Width, Slices, Volumes)
    # We take Slice 40, Volume 0
    slice_a = img1[:, :, 40, 0]
    slice_b = img2[:, :, 40, 0]

    # 4. Normalize (0 to 1) - Critical for AI
    def norm(x): return (x - np.min(x)) / (np.max(x) - np.min(x))
    
    input_tensor = torch.from_numpy(norm(slice_a)).unsqueeze(0).unsqueeze(0).float()
    target_tensor = torch.from_numpy(norm(slice_b)).unsqueeze(0).unsqueeze(0).float()

    # 5. Setup Model
    model = SimpleUNet()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    # 6. Training Loop
    print("Starting Training on Real Scans...")
    for epoch in range(201):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    # 7. Visualize Result
    with torch.no_grad():
        denoised = model(input_tensor).squeeze().numpy()

    plt.figure(figsize=(15, 5))
    plt.subplot(1,3,1); plt.imshow(np.rot90(slice_a), cmap='gray'); plt.title("Real Noisy Run 1")
    plt.subplot(1,3,2); plt.imshow(np.rot90(slice_b), cmap='gray'); plt.title("Real Noisy Run 2")
    plt.subplot(1,3,3); plt.imshow(np.rot90(denoised), cmap='gray'); plt.title("AI Denoised Result")
    plt.show()

if __name__ == "__main__":
    train_real_n2n()
    