"""Data loading and preprocessing for 3D SR."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


def center_crop_3d(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Extract the centered sub-volume with a given shape."""
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
    d, h, w = volume.shape
    td, th, tw = target_shape
    if d < td or h < th or w < tw:
        raise ValueError(f"Cannot crop volume {volume.shape} to {target_shape}")

    d0 = (d - td) // 2
    h0 = (h - th) // 2
    w0 = (w - tw) // 2
    return volume[d0 : d0 + td, h0 : h0 + th, w0 : w0 + tw]


def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    """Scale array values into [0, 1]."""
    arr = arr.astype(np.float32)
    amin = float(arr.min())
    amax = float(arr.max())
    if amax - amin < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - amin) / (amax - amin)


def resize_3d_numpy(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Resample a 3D NumPy volume to a target shape."""
    tensor = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()


class SRVolumeDataset(Dataset):
    """Dataset for paired degraded/gt 4D arrays saved as object npy lists."""

    def __init__(
        self,
        degraded_npy_path: Path,
        gt_npy_path: Path,
        input_patch_shape=(64, 64, 64),
        output_patch_shape=(128, 128, 128),
        samples_per_timepoint: int = 2,
        seed: int = 42,
    ):
        self.degraded_path = Path(degraded_npy_path)
        self.gt_path = Path(gt_npy_path)
        if not self.degraded_path.exists():
            raise FileNotFoundError(f"Degraded data file not found: {self.degraded_path}")
        if not self.gt_path.exists():
            raise FileNotFoundError(f"GT data file not found: {self.gt_path}")

        self.input_patch_shape = tuple(input_patch_shape)
        self.output_patch_shape = tuple(output_patch_shape)
        self.samples_per_timepoint = int(samples_per_timepoint)
        self.seed = int(seed)

        degraded_raw = np.load(self.degraded_path, allow_pickle=True)
        gt_raw = np.load(self.gt_path, allow_pickle=True)
        self.degraded_list = self._to_volume_list(degraded_raw, "degraded")
        self.gt_list = self._to_volume_list(gt_raw, "gt")

        if len(self.degraded_list) != len(self.gt_list):
            raise ValueError(
                f"Mismatching list lengths: degraded={len(self.degraded_list)} gt={len(self.gt_list)}"
            )

        self.paired_4d = []
        self.sample_index = []

        for vol_idx, (deg_4d, gt_4d) in enumerate(zip(self.degraded_list, self.gt_list)):
            if deg_4d.shape != gt_4d.shape:
                raise ValueError(f"Volume {vol_idx}: degraded {deg_4d.shape} != gt {gt_4d.shape}")
            if deg_4d.ndim != 4:
                raise ValueError(f"Volume {vol_idx} must be 4D (X,Y,Z,T), got {deg_4d.shape}")

            x, y, z, t = deg_4d.shape
            inx, iny, inz = self.input_patch_shape
            if x < inx or y < iny or z < inz:
                raise ValueError(
                    f"Volume {vol_idx} too small for input patch {self.input_patch_shape}: {deg_4d.shape}"
                )

            self.paired_4d.append((deg_4d.astype(np.float32), gt_4d.astype(np.float32)))
            for t_idx in range(t):
                for _ in range(self.samples_per_timepoint):
                    self.sample_index.append((vol_idx, t_idx))

        if len(self.sample_index) == 0:
            raise ValueError("No training samples built from degraded/gt data")

    def _to_volume_list(self, raw, name: str):
        if isinstance(raw, np.ndarray) and raw.dtype == object:
            values = raw.tolist()
        elif isinstance(raw, np.ndarray):
            if raw.ndim == 5:
                values = [raw[i] for i in range(raw.shape[0])]
            else:
                raise ValueError(f"{name} array has unsupported shape {raw.shape}")
        else:
            values = list(raw)

        parsed = []
        for idx, item in enumerate(values):
            if isinstance(item, (str, Path)):
                path_obj = Path(item)
                if not path_obj.exists():
                    raise FileNotFoundError(f"{name} volume path does not exist: {path_obj}")
                arr = np.load(path_obj, allow_pickle=True)
            else:
                arr = np.asarray(item)

            if arr.ndim != 4:
                raise ValueError(f"{name} volume {idx} must be 4D (X,Y,Z,T), got {arr.shape}")
            parsed.append(arr)
        return parsed

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        vol_idx, t_idx = self.sample_index[idx]
        deg_4d, gt_4d = self.paired_4d[vol_idx]

        deg_3d = normalize_minmax(deg_4d[:, :, :, t_idx])
        gt_3d = normalize_minmax(gt_4d[:, :, :, t_idx])

        inx, iny, inz = self.input_patch_shape
        x, y, z = deg_3d.shape

        rng = np.random.default_rng(self.seed + idx)
        x0 = int(rng.integers(0, x - inx + 1))
        y0 = int(rng.integers(0, y - iny + 1))
        z0 = int(rng.integers(0, z - inz + 1))

        x_patch = deg_3d[x0 : x0 + inx, y0 : y0 + iny, z0 : z0 + inz]
        y_patch_full = gt_3d[x0 : x0 + inx, y0 : y0 + iny, z0 : z0 + inz]
        y_patch = resize_3d_numpy(y_patch_full, self.output_patch_shape)

        x_tensor = torch.from_numpy(x_patch).unsqueeze(0)
        y_tensor = torch.from_numpy(y_patch).unsqueeze(0)
        return x_tensor, y_tensor


def create_dataloaders(config: dict):
    """Build train/validation data loaders from a config dict."""
    dataset = SRVolumeDataset(
        degraded_npy_path=config["data_file"],
        gt_npy_path=config["gt_file"],
        input_patch_shape=config["input_patch_shape"],
        output_patch_shape=config["output_patch_shape"],
        samples_per_timepoint=config["samples_per_timepoint"],
        seed=config["seed"],
    )

    train_size = max(1, int(len(dataset) * config["train_split"]))
    val_size = len(dataset) - train_size

    if val_size == 0 and len(dataset) > 1:
        val_size = 1
        train_size = len(dataset) - 1

    generator = torch.Generator().manual_seed(config["seed"])
    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )
    return train_loader, val_loader, len(dataset)
