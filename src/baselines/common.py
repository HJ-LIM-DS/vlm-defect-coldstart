"""비-VLM 기준선 공통 부품: 특징 결합, 인덱스 로드, 로지스틱 회귀, 예측 기록 저장.

특징 벡터 = [DINOv2 전체 이미지(1536), DINOv2 이상 영역 crop(1536), PatchCore 이미지 점수의 제품 내 z값(1)].
crop 특징과 PatchCore 점수는 test 이미지에만 있으므로 기준선은 test 결함 이미지로만 학습한다.
"""

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.baselines.features import load_features

FEATURE_DIR = Path("data/features")
PATCHCORE_DIR = Path("data/patchcore")


def load_index(path: Path = Path("data/index.csv")) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_defect_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["split"] == "test" and r["label"] == "defect"]


def patchcore_scores(rows: list[dict]) -> dict[str, float]:
    """제품별 scores.csv 를 모아 이미지 ID -> 점수. 제품 안에서 z값으로 바꾼다(제품마다 스케일이 다르다)."""
    out = {}
    for cat in sorted({r["category"] for r in rows}):
        path = PATCHCORE_DIR / cat / "scores.csv"
        with open(path, encoding="utf-8") as f:
            sub = {r["image_id"]: float(r["score"]) for r in csv.DictReader(f)}
        v = np.array(list(sub.values()))
        mu, sd = v.mean(), v.std() + 1e-8
        out.update({k: (s - mu) / sd for k, s in sub.items()})
    return out


class FeatureBank:
    def __init__(self, rows: list[dict], sources=("dinov2", "dinov2_crop"), with_score: bool = True):
        self.parts = []
        for name in sources:
            ids, feat = load_features(FEATURE_DIR / f"{name}.npz")
            self.parts.append({i: f for i, f in zip(ids, feat)})
        self.scores = patchcore_scores(rows) if with_score else None

    def vector(self, image_id: str) -> np.ndarray:
        v = [p[image_id] for p in self.parts]
        if self.scores is not None:
            v.append(np.array([self.scores[image_id]], dtype=np.float32))
        return np.concatenate(v)

    def matrix(self, image_ids: list[str]) -> np.ndarray:
        return np.stack([self.vector(i) for i in image_ids])


def fit_logreg(X: np.ndarray, y: list[str], C: float = 1.0, seed: int = 0):
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf.fit(scaler.transform(X), y)
    return scaler, clf


def predict_restricted(scaler, clf, X: np.ndarray, allowed: list[str]) -> list[str | None]:
    """클래스 확률 중 allowed(사전의 키)에 든 것만 비교한다. 분류기가 모르는 키뿐이면 None."""
    proba = clf.predict_proba(scaler.transform(X))
    classes = list(clf.classes_)
    cols = [classes.index(k) for k in allowed if k in classes]
    if not cols:
        return [None] * len(X)
    keys = [classes[c] for c in cols]
    return [keys[int(np.argmax(p[cols]))] for p in proba]


def write_predictions(path: Path, rows: list[dict], pred_keys: list, extra: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r, k in zip(rows, pred_keys):
            rec = {"image_id": r["image_id"], "category": r["category"], "label": r["label"],
                   "defect_type": r["defect_type"], "output": "", "pred_key": k, "pred_name": None,
                   "reason": None, "format_ok": k is not None}
            if extra:
                rec.update(extra)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
