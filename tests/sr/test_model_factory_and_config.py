import unittest

import torch

from src.sr.config import DEFAULT_CONFIG, validate_config
from src.sr.model import RCAN3D, SRCNN3D, build_model_from_config, select_model


class TestModelFactoryAndConfig(unittest.TestCase):
    def test_select_model_dispatches(self):
        srcnn = select_model("srcnn3d", output_patch_shape=(16, 16, 16))
        rcan = select_model("rcan3d", output_patch_shape=(24, 24, 24))
        self.assertIsInstance(srcnn, SRCNN3D)
        self.assertIsInstance(rcan, RCAN3D)

    def test_rcan3d_forward_matches_output_patch_shape(self):
        out_shape = (12, 14, 10)
        lr_shape = (6, 7, 5)
        model = select_model(
            "rcan3d",
            output_patch_shape=out_shape,
            n_resgroups=1,
            n_resblocks=1,
            n_feats=8,
        )
        x = torch.randn(2, 1, *lr_shape)
        y = model(x)
        self.assertEqual(tuple(y.shape), (2, 1, *out_shape))

    def test_select_model_unknown_raises(self):
        with self.assertRaises(ValueError):
            select_model("unknown-model")

    def test_build_model_from_config_uses_model_name_and_kwargs(self):
        config = dict(DEFAULT_CONFIG)
        config["model_name"] = "rcan3d"
        config["model_kwargs"] = {"output_patch_shape": (20, 20, 20)}
        model = build_model_from_config(config)
        self.assertIsInstance(model, RCAN3D)
        self.assertEqual(model.output_patch_shape, (20, 20, 20))

    def test_validate_config_rejects_unknown_model(self):
        config = dict(DEFAULT_CONFIG)
        config["model_name"] = "not-real"
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_validate_config_rejects_unknown_loss(self):
        config = dict(DEFAULT_CONFIG)
        config["loss_name"] = "not-real"
        with self.assertRaises(ValueError):
            validate_config(config)

if __name__ == "__main__":
    unittest.main()
