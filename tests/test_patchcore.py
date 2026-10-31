"""PatchCore 보조 함수 검사(백본 로드 없이)."""

import numpy as np
import pytest
from PIL import Image

from src.detector.patchcore import CROP_FRAC, crop_at_peak, greedy_coreset


def test_crop_follows_peak_and_stays_inside(tmp_path):
    W, H = 900, 600
    img = Image.new("RGB", (W, H), (10, 10, 10))
    p = tmp_path / "img.png"
    img.save(p)
    amap = np.zeros((H, W), np.float32)
    amap[5, 890] = 1.0  # 우상단 모서리 근처
    crop = crop_at_peak(p, amap)
    side = int(round(max(W, H) * CROP_FRAC))
    assert crop.size == (side, side)
    amap[:] = 0
    amap[300, 450] = 1.0  # 중앙
    crop = crop_at_peak(p, amap)
    assert crop.size == (side, side)


def test_coreset_picks_spread_points():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(0, 0.01, size=(500, 32))
    cluster_b = rng.normal(5, 0.01, size=(500, 32))
    feats = np.concatenate([cluster_a, cluster_b]).astype(np.float32)
    idx = greedy_coreset(feats, ratio=0.01, device="cpu")
    assert len(idx) == 10
    assert len(set(idx)) == 10
    assert any(i < 500 for i in idx) and any(i >= 500 for i in idx)
