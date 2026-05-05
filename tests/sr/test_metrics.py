"""Tests for SR training metrics (no manifest / I/O)."""

import unittest

import torch

from src.sr.training import masked_local_ssim_3d, masked_mse_loss


class TestMaskedMetrics(unittest.TestCase):
    def test_masked_mse_zero_for_identical(self):
        x = torch.rand(2, 1, 8, 8, 8)
        m = torch.ones_like(x)
        loss = masked_mse_loss(x, x, m)
        self.assertLess(loss.item(), 1e-6)

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
