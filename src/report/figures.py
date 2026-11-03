"""집계 표(reports/tables/summary.json)에서 보고용 그림을 만든다(reports/figures/*.png, *.svg).

그림
- zero_label: 새 제품 라벨 0장에서 방법별 유형 정확도(제품 평균). 적응 VLM, zero-shot VLM, 비-VLM을 색으로 가른다.
- pairs: 짝지은 비교의 제품 평균 차이와 95% 쌍대 구간. 0을 가로지르는지 한눈에 보인다.
- crossover: 유형당 k장 분류기의 정확도 곡선과 라벨 0장 VLM의 수평선. 교차점 k를 읽는다.
- scope: 공통 유형과 고유 유형으로 나눈 정확도. 적응과 정의 문장의 이득이 어느 쪽에서 오는지 보인다.
- per_product: 제품별 정확도 열 지도. fold 간 분산과 어려운 제품을 보인다.

색은 네 가지 역할(적응 VLM, zero-shot VLM, 비-VLM 라벨 0장, few-shot 분류기)에 고정하고 순서를 바꾸지 않는다.
글자는 한국어 글꼴(Malgun Gothic)이 있으면 쓰고 없으면 기본 글꼴로 떨어진다.

사용법:
    python -m src.report.figures
    python -m src.report.figures --summary reports/tables/summary.json --out-dir reports/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

COLOR = {"adapted": "#2a78d6", "zeroshot": "#eb6834", "nonvlm": "#1baf7a", "fewshot": "#eda100", "neutral": "#9a9a94"}
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# 그림에 쓰는 방법 이름과 색 역할. 표의 내부 이름을 보고서의 기호로 바꾼다.
METHODS = {
    "b_cross": ("B-교차", "nonvlm"), "b_clip": ("B-CLIP", "nonvlm"),
    "z2b": ("Z-2B (2B zero-shot)", "zeroshot"), "z4b_noref": ("Z (4B zero-shot)", "zeroshot"),
    "2b_noref": ("2B LoRA", "adapted"), "2b_nodef": ("2B LoRA, 정의 없음", "adapted"), "l4b": ("L (4B QLoRA)", "adapted"),
    "b_kshot_k1": ("B-1장", "fewshot"), "b_full": ("B-전체", "fewshot"),
}


def setup_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "Apple SD Gothic Neo"):
        if cand in names:
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2, "ytick.color": TEXT2,
                         "text.color": TEXT, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def label(name: str) -> str:
    return METHODS.get(name, (name, "neutral"))[0]


def color(name: str) -> str:
    return COLOR[METHODS.get(name, (name, "neutral"))[1]]


def save(fig, out_dir: Path, stem: str):
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_zero_label(summary: dict, out_dir: Path):
    order = ["b_cross", "z2b", "b_clip", "z4b_noref", "2b_noref", "l4b"]
    vals = [100 * summary[m]["type_acc_product_mean"] for m in order]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(order))
    ax.barh(y, vals, height=0.62, color=[color(m) for m in order])
    for yi, v in zip(y, vals):
        ax.text(v + 0.8, yi, f"{v:.1f}", va="center", ha="left", color=TEXT, fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels([label(m) for m in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 70)
    ax.set_xlabel("유형 정확도, 제품 평균 (%)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("새 제품 라벨 0장: 결함 사전만으로 유형을 맞힌 비율", loc="left", fontsize=11, color=TEXT)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[k]) for k in ("adapted", "zeroshot", "nonvlm")]
    ax.legend(handles, ["LoRA 적응 VLM", "zero-shot VLM", "비-VLM 기준선"], loc="upper right", frameon=False, fontsize=9)
    save(fig, out_dir, "zero_label")


def fig_pairs(pairs: dict, out_dir: Path):
    rows = [(k, v) for k, v in pairs.items() if k.split(" 대 ")[0] in METHODS and k.split(" 대 ")[1] in METHODS]
    rows.sort(key=lambda kv: kv[1]["diff"])
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(rows) + 1.4))
    y = np.arange(len(rows))
    for yi, (name, p) in zip(y, rows):
        a, b = name.split(" 대 ")
        lo, hi = 100 * p["ci95"][0], 100 * p["ci95"][1]
        c = color(a) if p["diff"] >= 0 else COLOR["neutral"]
        ax.plot([lo, hi], [yi, yi], color=c, linewidth=2, solid_capstyle="round")
        ax.plot(100 * p["diff"], yi, "o", color=c, markersize=8, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(hi + 1.2, yi, f"{100 * p['diff']:+.1f} [{lo:+.1f}, {hi:+.1f}]", va="center", ha="left", color=TEXT2, fontsize=9)
    ax.axvline(0, color=TEXT2, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{label(a)} 대 {label(b)}" for a, b in (k.split(" 대 ") for k, _ in rows)])
    ax.set_xlabel("제품 평균 정확도 차이 (A 빼기 B, %p)와 95% 쌍대 구간")
    lo_all = min(100 * p["ci95"][0] for _, p in rows)
    hi_all = max(100 * p["ci95"][1] for _, p in rows)
    ax.set_xlim(lo_all - 4, hi_all + 22)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("짝지은 비교: 같은 이미지에서 두 방법의 정확도 차이", loc="left", fontsize=11, color=TEXT)
    save(fig, out_dir, "pairs")


def fig_crossover(summary: dict, out_dir: Path):
    ks = sorted(int(k[len("b_kshot_k"):]) for k in summary if k.startswith("b_kshot_k"))
    mean = np.array([100 * summary[f"b_kshot_k{k}"]["type_acc_product_mean"] for k in ks])
    sd = np.array([100 * summary[f"b_kshot_k{k}"]["type_acc_product_std"] for k in ks])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(ks)) + 1
    ax.fill_between(x, mean - sd, mean + sd, color=COLOR["fewshot"], alpha=0.18, linewidth=0)
    ax.plot(x, mean, color=COLOR["fewshot"], linewidth=2, marker="o", markersize=7, markeredgecolor="white")
    for xi, m in zip(x, mean):
        ax.text(xi, m + 2.2, f"{m:.1f}", ha="center", color=TEXT, fontsize=9)
    for name in ("l4b", "z4b_noref", "b_clip"):
        v = 100 * summary[name]["type_acc_product_mean"]
        ax.plot(0, v, "o", color=color(name), markersize=8, markeredgecolor="white", markeredgewidth=1.5)
        ax.hlines(v, 0, x[-1], color=color(name), linewidth=1.2, linestyle=(0, (4, 3)))
        ax.text(x[-1] + 0.08, v, f"{label(name)} {v:.1f}", va="center", ha="left", color=TEXT, fontsize=9)
    ax.set_xticks([0, *x])
    ax.set_xticklabels(["0", *[str(k) for k in ks]])
    ax.set_xlim(-0.3, x[-1] + 1.9)
    ax.set_ylim(15, 95)
    ax.set_xlabel("새 제품 결함 유형당 라벨 수 k (B-k장 분류기, 20회 평균과 표준편차)")
    ax.set_ylabel("유형 정확도, 제품 평균 (%)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("교차점: 라벨 몇 장이면 분류기가 라벨 0장 VLM을 넘는가", loc="left", fontsize=11, color=TEXT)
    save(fig, out_dir, "crossover")


def fig_scope(summary: dict, out_dir: Path):
    order = ["b_clip", "z2b", "z4b_noref", "2b_nodef", "2b_noref", "l4b", "b_kshot_k1"]
    shared = [100 * summary[m]["type_acc_shared"] for m in order]
    unique = [100 * summary[m]["type_acc_unique"] for m in order]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    x = np.arange(len(order))
    w = 0.38
    ax.bar(x - w / 2, shared, width=w - 0.03, color=COLOR["adapted"], label="공통 유형 (두 제품 이상에 있는 키)")
    ax.bar(x + w / 2, unique, width=w - 0.03, color=COLOR["zeroshot"], label="고유 유형 (한 제품에만 있는 키)")
    for xi, s, u in zip(x, shared, unique):
        ax.text(xi - w / 2, s + 1, f"{s:.0f}", ha="center", color=TEXT, fontsize=8.5)
        ax.text(xi + w / 2, u + 1, f"{u:.0f}", ha="center", color=TEXT, fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([label(m).replace(" (", "\n(").replace(", ", "\n") for m in order], fontsize=8.5)
    ax.set_ylim(0, 85)
    ax.set_ylabel("유형 정확도, 제품 평균 (%)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("공통 유형과 고유 유형: 적응의 이득은 어디서 오는가", loc="left", fontsize=11, color=TEXT)
    save(fig, out_dir, "scope")


def fig_per_product(summary: dict, out_dir: Path):
    order = ["b_clip", "z2b", "z4b_noref", "2b_noref", "l4b", "b_kshot_k1", "b_full"]
    cats = sorted(summary["l4b"]["per_product"])
    mat = np.array([[100 * summary[m]["per_product"].get(c, np.nan) for m in order] for c in cats])
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    im = ax.imshow(mat, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#cde2fb", "#6da7ec", "#2a78d6", "#104281"]),
                   vmin=0, vmax=100, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5, color="white" if v > 60 else TEXT)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label(m).replace(" (", "\n(") for m in order], fontsize=8.5)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("유형 정확도 (%)", color=TEXT2)
    cb.outline.set_visible(False)
    ax.set_title("제품별 유형 정확도 (toothbrush는 유형이 하나라 제외)", loc="left", fontsize=11, color=TEXT)
    save(fig, out_dir, "per_product")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, default=Path("reports/tables/summary.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    args = ap.parse_args()
    setup_font()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    summary, pairs = data["summary"], data.get("pairs", {})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_zero_label(summary, args.out_dir)
    if pairs:
        fig_pairs(pairs, args.out_dir)
    fig_crossover(summary, args.out_dir)
    fig_scope(summary, args.out_dir)
    fig_per_product(summary, args.out_dir)
    print(f"저장: {args.out_dir}")


if __name__ == "__main__":
    main()
