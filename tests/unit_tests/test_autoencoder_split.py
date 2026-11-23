import numpy as np
import pytest
import torch

pytest.importorskip("legged_fld")
from scripts.train_autoencoder import (
    KINE2GO_JOINT_COLS,
    build_model,
    fld_loss,
    load_split,
    sequences_from_clips,
    windows_from_clips
)

H = 61
N = 5


def write_clip(tmp_path, name, joints):
    """Save a (T, 61) Kine2Go array whose 12 joint columns hold `joints` (T, 12)."""
    state = np.zeros((len(joints), 61), dtype=np.float32)
    state[:, KINE2GO_JOINT_COLS] = joints
    path = tmp_path / f"{name}.npy"
    np.save(path, state)
    return str(path)


def test_no_window_mixes_clips():
    a, b = torch.zeros(100, 12), torch.ones(80, 12)
    x = windows_from_clips([a, b], torch.zeros(12), torch.ones(12), H)
    assert x.shape[0] == (100 - H + 1) + (80 - H + 1)
    # each clip is constant, so a window inside one clip must be constant.
    assert torch.all(x.std(dim=-1) == 0)


def test_short_clip_raises():
    with pytest.raises(ValueError):
        windows_from_clips([torch.zeros(H - 1, 12)], torch.zeros(12), torch.ones(12), H)


def test_test_clip_is_not_in_train_data(tmp_path):
    train = write_clip(tmp_path, "walk_a", np.zeros((100, 12)))
    val = write_clip(tmp_path, "walk_b", np.full((100, 12), 1e6))
    train_trajs, val_trajs = load_split([train], [val], 1.0, H)
    assert list(val_trajs) == ["walk_b"]
    assert torch.cat(train_trajs).abs().max() == 0


def test_fraction_split_is_per_clip_and_skips_short(tmp_path):
    long = write_clip(tmp_path, "long", np.zeros((400, 12)))
    short = write_clip(tmp_path, "short", np.zeros((200, 12)))
    train_trajs, val_trajs = load_split([long, short], [], 0.8, H)
    assert [len(t) for t in train_trajs] == [320, 160]
    assert list(val_trajs) == ["long"]

def test_no_sequence_mixes_clips():
    a, b = torch.zeros(100, 12), torch.ones(80, 12)
    seq = sequences_from_clips([a, b], torch.zeros(12), torch.ones(12), H, N)
    assert seq.shape == ((100 - H - N + 1) + (80 - H - N + 1), N + 1, 12, H)
    # each clip is constant, so each sequence is constant over steps and time
    assert torch.all(seq.flatten(1).std(dim=-1) == 0)


def test_chunked_fld_loss_is_exact():
    torch.manual_seed(0)
    model = build_model("fld", 12, 3, 1).eval()
    seq = torch.randn(7, N + 1, 12, H)
    with torch.no_grad():
        full = model.loss(seq).item()
    # 7 = 3 + 3 + 1: unequal chunks, so a plain mean of chunk losses would be wrong
    assert fld_loss(model, seq, 1.0, chunk=3) == pytest.approx(full, rel=1e-5)

