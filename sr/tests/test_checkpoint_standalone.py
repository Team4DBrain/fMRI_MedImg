"""Tests for standalone checkpoint config resolution."""

from __future__ import annotations

from pathlib import Path

import torch

from sr.checkpoint import (
    EpochState,
    capture_rng_state,
    embed_config_in_checkpoint,
    find_config_json_for_checkpoint,
    load_config_for_inference,
    load_epoch,
    resolve_checkpoint_for_model,
)
from sr.config import SRConfig, to_json
from sr.infer import _load_model_from_checkpoint
from sr.models import build_model


def test_find_sidecar_config_next_to_checkpoint(tmp_path: Path) -> None:
    ckpt = tmp_path / "model_best.pt"
    ckpt.write_bytes(b"placeholder")
    sidecar = tmp_path / "model_best.config.json"
    config = SRConfig(model_name="srcnn3d")
    to_json(config, sidecar)

    found = find_config_json_for_checkpoint(ckpt)
    assert found == sidecar.resolve()
    loaded = load_config_for_inference(ckpt, model_name="srcnn3d")
    assert loaded.model_name == "srcnn3d"


def test_resolve_checkpoint_without_run_dir(tmp_path: Path) -> None:
    ckpt = tmp_path / "rcan3d_best.pt"
    config = SRConfig(
        model_name="rcan3d",
        model_kwargs={"n_feats": 16, "n_resgroups": 1, "n_resblocks": 1, "reduction": 8},
        output_patch_shape=(16, 16, 12),
    )
    to_json(config, tmp_path / "rcan3d_best.config.json")

    model = build_model(config)
    state = EpochState(
        epoch_number=1,
        model_state_dict=model.state_dict(),
        optimizer_state_dict={},
        scheduler_state_dict=None,
        rng_state=capture_rng_state(),
        metrics_history=[],
        best_val_loss=0.0,
        best_epoch_number=1,
    )
    torch.save(
        {
            "epoch_number": state.epoch_number,
            "model_state_dict": state.model_state_dict,
            "optimizer_state_dict": state.optimizer_state_dict,
            "scheduler_state_dict": state.scheduler_state_dict,
            "rng_state": state.rng_state,
            "metrics_history": state.metrics_history,
            "best_val_loss": state.best_val_loss,
            "best_epoch_number": state.best_epoch_number,
            "loss_name": state.loss_name,
            "extra": state.extra,
        },
        ckpt,
    )

    resolved = resolve_checkpoint_for_model(
        "rcan3d",
        checkpoint=ckpt,
    )
    assert resolved == ckpt.resolve()

    loaded_model, loaded_config, _device = _load_model_from_checkpoint(
        ckpt,
        model_name="rcan3d",
    )
    assert loaded_config.model_name == "rcan3d"
    assert loaded_model is not None


def test_embedded_config_in_checkpoint_payload(tmp_path: Path) -> None:
    config = SRConfig(
        model_name="srcnn3d",
        output_patch_shape=(16, 16, 12),
    )
    model = build_model(config)
    ckpt = tmp_path / "weights.pt"
    payload = {
        "epoch_number": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": None,
        "rng_state": {},
        "metrics_history": [],
        "best_val_loss": 0.0,
        "best_epoch_number": 1,
        "loss_name": "masked_mse",
        "extra": {
            "config": {
                "model_name": "srcnn3d",
                "output_patch_shape": [16, 16, 12],
                "manifest_path": str(config.manifest_path),
                "run_root": str(config.run_root),
                "patch_hr_shape": list(config.patch_hr_shape),
            }
        },
    }
    torch.save(payload, ckpt)

    loaded = load_config_for_inference(ckpt, model_name="srcnn3d")
    assert loaded.model_name == "srcnn3d"
    assert tuple(loaded.output_patch_shape) == (16, 16, 12)
    state = load_epoch(ckpt)
    assert state.model_state_dict


def test_embed_config_in_checkpoint_round_trip(tmp_path: Path) -> None:
    config = SRConfig(
        model_name="srcnn3d",
        output_patch_shape=(16, 16, 12),
    )
    model = build_model(config)
    ckpt = tmp_path / "weights.pt"
    to_json(config, tmp_path / "weights.config.json")
    payload = {
        "epoch_number": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": None,
        "rng_state": {},
        "metrics_history": [],
        "best_val_loss": 0.0,
        "best_epoch_number": 1,
        "loss_name": "masked_mse",
        "extra": {},
    }
    torch.save(payload, ckpt)

    embed_config_in_checkpoint(ckpt, backup=True, dry_run=False)
    assert (tmp_path / "weights.pt.bak").is_file()

    loaded = load_config_for_inference(ckpt, model_name="srcnn3d")
    assert loaded.model_name == "srcnn3d"
    assert tuple(loaded.output_patch_shape) == (16, 16, 12)

    reloaded = load_epoch(ckpt)
    model2 = build_model(loaded)
    model2.load_state_dict(reloaded.model_state_dict, strict=True)


def test_embed_config_refuses_without_sidecar(tmp_path: Path) -> None:
    config = SRConfig(model_name="srcnn3d")
    model = build_model(config)
    ckpt = tmp_path / "orphan.pt"
    torch.save(
        {
            "epoch_number": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": None,
            "rng_state": {},
            "metrics_history": [],
            "best_val_loss": 0.0,
            "best_epoch_number": 1,
            "loss_name": "masked_mse",
            "extra": {},
        },
        ckpt,
    )
    import pytest

    with pytest.raises(FileNotFoundError):
        embed_config_in_checkpoint(ckpt)


def test_embed_config_dry_run_on_already_embedded(tmp_path: Path) -> None:
    config = SRConfig(
        model_name="srcnn3d",
        output_patch_shape=(16, 16, 12),
    )
    model = build_model(config)
    ckpt = tmp_path / "weights.pt"
    from sr.config import _config_to_dict

    payload = {
        "epoch_number": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": {},
        "scheduler_state_dict": None,
        "rng_state": {},
        "metrics_history": [],
        "best_val_loss": 0.0,
        "best_epoch_number": 1,
        "loss_name": "masked_mse",
        "extra": {"config": _config_to_dict(config)},
    }
    torch.save(payload, ckpt)

    embed_config_in_checkpoint(ckpt, dry_run=True)
