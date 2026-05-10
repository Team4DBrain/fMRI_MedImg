"""Tests for SR training metrics (no manifest / I/O)."""

import unittest

import torch

from src.sr.training import (
    l1_loss,
    masked_l1_loss,
    masked_local_ssim_3d,
    masked_mse_loss,
    mse_loss,
    resolve_loss_function,
)


class TestMaskedMetrics(unittest.TestCase):
    def test_masked_mse_zero_for_identical(self):
        x = torch.rand(2, 1, 8, 8, 8)
        m = torch.ones_like(x)
        loss = masked_mse_loss(x, x, m)
        self.assertLess(loss.item(), 1e-6)

    def test_unmasked_mse_includes_background(self):
        pred = torch.tensor([[[[[2.0, 4.0]]]]])
        target = torch.zeros_like(pred)
        mask = torch.tensor([[[[[1.0, 0.0]]]]])

        self.assertAlmostEqual(masked_mse_loss(pred, target, mask).item(), 4.0)
        self.assertAlmostEqual(mse_loss(pred, target, mask).item(), 10.0)

    def test_masked_and_unmasked_l1(self):
        pred = torch.tensor([[[[[2.0, -4.0]]]]])
        target = torch.zeros_like(pred)
        mask = torch.tensor([[[[[1.0, 0.0]]]]])

        self.assertAlmostEqual(masked_l1_loss(pred, target, mask).item(), 2.0)
        self.assertAlmostEqual(l1_loss(pred, target, mask).item(), 3.0)

    def test_resolve_loss_function_rejects_unknown_name(self):
        self.assertIs(resolve_loss_function("masked_l1"), masked_l1_loss)
        with self.assertRaises(ValueError):
            resolve_loss_function("not-a-loss")

    def test_ssim_one_for_identical(self):
        x = torch.rand(1, 1, 16, 16, 16)
        m = torch.ones_like(x)
        s = masked_local_ssim_3d(x, x, m)
        self.assertGreater(s.item(), 0.99)

    def test_ssim_in_valid_range(self):
        pred = torch.rand(1, 1, 12, 12, 12)
        tgt = torch.rand(1, 1, 12, 12, 12)
        m = torch.ones_like(pred)
        s = masked_local_ssim_3d(pred, tgt, m)
        self.assertGreaterEqual(s.item(), -1.0)
        self.assertLessEqual(s.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
