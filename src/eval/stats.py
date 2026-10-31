"""짝지은(같은 이미지) 비교 통계.

- product_mean: 제품별 정확도의 평균. 제품마다 장수가 달라 이미지 평균은 큰 제품에 끌리므로 주 지표는 이 값이다.
- paired_bootstrap: 두 방법의 정오 벡터를 같은 이미지로 짝지어, 제품 안에서 이미지를 복원 추출하는 층화 부트스트랩으로
  제품 평균 정확도 차이의 95% 구간을 구한다.
- mcnemar: 불일치 쌍에 대한 정확 이항 검정.
- holm: 여러 비교의 p값 보정.
"""

import numpy as np
from scipy.stats import binomtest


def product_mean(correct: np.ndarray, groups: np.ndarray) -> float:
    correct, groups = np.asarray(correct, float), np.asarray(groups)
    return float(np.mean([correct[groups == g].mean() for g in np.unique(groups)]))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, groups: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    """제품 평균 정확도 차이(a - b)의 점추정과 95% 백분위 구간."""
    a, b, groups = np.asarray(a, float), np.asarray(b, float), np.asarray(groups)
    if not (len(a) == len(b) == len(groups)):
        raise ValueError("길이가 다르다")
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    diffs = np.zeros(n_boot)
    for g in uniq:
        ag, bg = a[groups == g], b[groups == g]
        idx = rng.integers(0, len(ag), size=(n_boot, len(ag)))
        diffs += ag[idx].mean(1) - bg[idx].mean(1)
    diffs /= len(uniq)
    return {
        "diff": product_mean(a, groups) - product_mean(b, groups),
        "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
        "n": int(len(a)), "n_groups": int(len(uniq)),
    }


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = 1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue)
    return {"n10": n10, "n01": n01, "p": p}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


def format_p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"
