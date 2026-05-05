import tempfile
import unittest
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.sr.config import DEFAULT_CONFIG
from src.sr.data import create_dataloaders
from src.sr.model import build_model_from_config
from src.sr.training import ensure_finite_loss, save_checkpoint, train_one_epoch


class TestReproducibilityAndSafety(unittest.TestCase):
    def _make_config_and_data(self, root: Path) -> dict:
        rng = np.random.default_rng(7)
        bids_root = root / "bids"
        derivatives_dir = root / "derivatives" / "masks"
        bids_root.mkdir(parents=True, exist_ok=True)
        derivatives_dir.mkdir(parents=True, exist_ok=True)

        runs = []
        for subject in ("01", "02"):
            vol = rng.normal(size=(10, 10, 10, 4)).astype(np.float32)
            img_path = bids_root / f"sub-{subject}_ses-00_task-Test_dir-ap_bold.nii.gz"
            nib.save(nib.Nifti1Image(vol, affine=np.eye(4)), str(img_path))

            mask = np.ones((10, 10, 10), dtype=np.uint8)
            mask_path = derivatives_dir / f"sub-{subject}_ses-00_task-Test_dir-ap_mask.nii.gz"
            nib.save(nib.Nifti1Image(mask, affine=np.eye(4)), str(mask_path))

            runs.append(
                {
                    "run_id": f"sub-{subject}_ses-00_task-Test_dir-ap",
                    "subject": subject,
                    "path": img_path.name,
                    "n_volumes": int(vol.shape[3]),
                    "norm_ref": 1.0,
                    "mask_path": f"masks/{mask_path.name}",
                }
            )

        manifest_path = root / "manifest.json"
        manifest = {
            "pipeline": "no_crop_v1",
            "bids_root": str(bids_root),
            "derivatives_dir": str(root / "derivatives"),
            "target_shape": [10, 10, 10],
            "target_z": 10,
            "runs": runs,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        config = dict(DEFAULT_CONFIG)
        config.update(
            {
                "manifest_path": manifest_path,
                "output_patch_shape": (10, 10, 10),
                "batch_size": 2,
                "num_workers": 0,
                "train_subjects": ["01"],
                "val_subjects": ["02"],
                "deterministic": True,
                "source_voxel_mm": 1.5,
                "target_voxel_mm": 3.0,
            }
        )
        return config

    def test_dataloader_seeded_order_is_stable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = self._make_config_and_data(root)

            train_loader_a, _, _, _ = create_dataloaders(config)
            train_loader_b, _, _, _ = create_dataloaders(config)

            batch_a = next(iter(train_loader_a))["input"]
            batch_b = next(iter(train_loader_b))["input"]
            self.assertTrue(torch.allclose(batch_a, batch_b))

    def test_finite_loss_guard_raises_on_nan(self):
        with self.assertRaises(FloatingPointError):
            ensure_finite_loss(torch.tensor(float("nan")), epoch_index=0, batch_idx=0)

    def test_checkpoint_write_is_atomic_and_loadable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = self._make_config_and_data(root)
            model = build_model_from_config(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            ckpt_path = root / "epoch_001.pt"

            save_checkpoint(ckpt_path, epoch=0, model=model, optimizer=optimizer, best_val_loss=0.1, config=config)

            self.assertTrue(ckpt_path.exists())
            self.assertFalse((root / "epoch_001.pt.tmp").exists())
            payload = torch.load(ckpt_path, map_location="cpu")
            self.assertIn("model_state_dict", payload)
            self.assertEqual(payload["epoch"], 0)

    def test_train_one_epoch_can_disable_finite_guard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = self._make_config_and_data(root)
            model = build_model_from_config(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            train_loader, _, _, _ = create_dataloaders(config)
            writer = SummaryWriter(log_dir=str(root / "tb"))
            try:
                loss_value = train_one_epoch(
                    0,
                    model,
                    train_loader,
                    optimizer,
                    "cpu",
                    writer,
                    strict_finite_loss=False,
                )
            finally:
                writer.close()
            self.assertTrue(np.isfinite(loss_value))


if __name__ == "__main__":
    unittest.main()
