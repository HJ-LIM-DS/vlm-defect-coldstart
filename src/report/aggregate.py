"""예측 기록을 모아 지표 표를 만든다(reports/tables/summary.json, summary.md, per_product.md).

입력
- reports/predictions/*.jsonl: 비-VLM 기준선(src.baselines.*). b_kshot_k{k}_rep{r}.jsonl 은 반복의 평균과 표준편차로 묶는다.
- --vlm 이름=패턴: VLM 결과. fold별 예측 파일을 이어 붙여 15개 제품을 한 번에 채점한다.
- --pair A:B: 두 방법을 같은 이미지로 짝지어 제품 평균 정확도 차이의 부트스트랩 구간과 McNemar 검정을 구한다.
  여러 쌍의 p값은 Holm 보정한다.
- --pair-scope A:B: 같은 비교를 공통 유형, 고유 유형, 정상 판정률로 나누어 구한다(정의 문장 효과와 정상 판정 가설용).

주 지표는 유형 정확도의 제품 평균(유형이 하나뿐인 toothbrush 제외 14개)이다. 정의는 src.eval.metrics 를 따른다.

사용법:
    python -m src.report.aggregate
    python -m src.report.aggregate --vlm z4b=runs/z4b/f*_predictions.jsonl --vlm 2b_full=runs/2b_full/f*/eval_predictions.jsonl
    python -m src.report.aggregate --vlm 2b_full=... --pair 2b_full:b_cross --pair 2b_full:b_clip
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.lexicon import load_lexicon, type_scope
from src.eval.metrics import evaluate, single_type_products
from src.eval.stats import format_p, holm, mcnemar, paired_bootstrap, product_mean

PRED_DIR = Path("reports/predictions")
TABLE_DIR = Path("reports/tables")
COLUMNS = [("type_acc_product_mean", "유형 정확도(제품 평균)"), ("type_acc_image_mean", "이미지 평균"),
           ("type_acc_shared", "공통 유형"), ("type_acc_unique", "고유 유형"), ("macro_f1", "macro-F1"),
           ("normal_rate", "정상 판정률"), ("format_rate", "형식 준수율")]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def load_glob(pattern: str) -> list[dict]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"파일 없음: {pattern}")
    rows = []
    for f in files:
        rows.extend(load_jsonl(Path(f)))
    return rows


def pct(x) -> str:
    return "" if x is None or (isinstance(x, float) and x != x) else f"{100 * x:.1f}"


def baseline_methods() -> dict[str, list[dict]]:
    methods = {}
    for p in sorted(PRED_DIR.glob("*.jsonl")):
        if re.match(r"b_kshot_k\d+_rep\d+$", p.stem):
            continue
        methods[p.stem] = load_jsonl(p)
    return methods


def kshot_summary(scope: dict) -> dict:
    by_k = defaultdict(list)
    for p in PRED_DIR.glob("b_kshot_k*_rep*.jsonl"):
        m = re.match(r"b_kshot_k(\d+)_rep(\d+)$", p.stem)
        by_k[int(m.group(1))].append(evaluate(load_jsonl(p), scope))
    out = {}
    for k, evs in sorted(by_k.items()):
        acc = [e["type_acc_product_mean"] for e in evs]
        cats = sorted({c for e in evs for c in e["per_product"]})
        out[f"b_kshot_k{k}"] = {
            "k": k, "reps": len(evs),
            "type_acc_product_mean": float(np.mean(acc)), "type_acc_product_std": float(np.std(acc)),
            "type_acc_image_mean": float(np.mean([e["type_acc_image_mean"] for e in evs])),
            "type_acc_shared": float(np.mean([e["type_acc_shared"] for e in evs])),
            "type_acc_unique": float(np.mean([e["type_acc_unique"] for e in evs])),
            "macro_f1": float(np.mean([e["macro_f1"] for e in evs])),
            "n_defect": float(np.mean([e["n_defect"] for e in evs])),
            "per_product": {c: float(np.mean([e["per_product"][c] for e in evs if c in e["per_product"]])) for c in cats},
        }
    return out


def defect_map(rows: list[dict], scope: dict) -> dict[str, tuple[float, str]]:
    skip = single_type_products(scope)
    return {r["image_id"]: (float(r["pred_key"] == r["defect_type"]), r["category"])
            for r in rows if r["label"] == "defect" and r["category"] not in skip}


def pairwise(rows_a: list[dict], rows_b: list[dict], scope: dict, n_boot: int) -> dict:
    """같은 이미지(결함, 유형이 하나뿐인 제품 제외)로 짝지은 비교. 차이는 a - b."""
    a, b = defect_map(rows_a, scope), defect_map(rows_b, scope)
    ids = sorted(set(a) & set(b))
    if not ids:
        raise SystemExit("겹치는 이미지가 없다")
    ca = np.array([a[i][0] for i in ids])
    cb = np.array([b[i][0] for i in ids])
    g = np.array([a[i][1] for i in ids])
    out = paired_bootstrap(ca, cb, g, n_boot=n_boot)
    out.update(mcnemar(ca, cb))
    out["acc_a"], out["acc_b"] = product_mean(ca, g), product_mean(cb, g)
    return out


def pairwise_scope(rows_a: list[dict], rows_b: list[dict], scope: dict, n_boot: int) -> dict:
    """공통 유형, 고유 유형, 정상 이미지로 나누어 짝지은 비교. 정의 문장 효과(H3)와 정상 판정률(H5)의 판정에 쓴다.

    유형 범위는 결함 이미지를 (제품, 정답 키)의 공통/고유로 가르고, 정상은 'good' 예측의 정오를 비교한다.
    """
    skip = single_type_products(scope)
    ia = {r["image_id"]: r for r in rows_a}
    ib = {r["image_id"]: r for r in rows_b}
    ids = sorted(set(ia) & set(ib))
    out = {}
    for name in ("shared", "unique"):
        sel = [i for i in ids if ia[i]["label"] == "defect" and ia[i]["category"] not in skip
               and scope.get((ia[i]["category"], ia[i]["defect_type"])) == name]
        ca = np.array([ia[i]["pred_key"] == ia[i]["defect_type"] for i in sel], float)
        cb = np.array([ib[i]["pred_key"] == ib[i]["defect_type"] for i in sel], float)
        g = np.array([ia[i]["category"] for i in sel])
        r = paired_bootstrap(ca, cb, g, n_boot=n_boot)
        r.update(mcnemar(ca, cb))
        r["acc_a"], r["acc_b"] = product_mean(ca, g), product_mean(cb, g)
        out[name] = r
    sel = [i for i in ids if ia[i]["label"] == "good"]
    ca = np.array([ia[i]["pred_key"] == "good" for i in sel], float)
    cb = np.array([ib[i]["pred_key"] == "good" for i in sel], float)
    g = np.array([ia[i]["category"] for i in sel])
    r = paired_bootstrap(ca, cb, g, n_boot=n_boot)
    r.update(mcnemar(ca, cb))
    r["acc_a"], r["acc_b"] = product_mean(ca, g), product_mean(cb, g)
    out["normal"] = r
    return out


SCOPE_LABEL = {"shared": "공통 유형", "unique": "고유 유형", "normal": "정상 판정률"}


def summary_markdown(summary: dict, pairs: dict, pairs_scope: dict | None = None) -> str:
    lines = ["| 방법 | " + " | ".join(c[1] for c in COLUMNS) + " | 결함 장수 |", "|---" * (len(COLUMNS) + 2) + "|"]
    for name, s in summary.items():
        cells = []
        for key, _ in COLUMNS:
            v = pct(s.get(key))
            if key == "type_acc_product_mean" and "type_acc_product_std" in s:
                v += f" (표준편차 {pct(s['type_acc_product_std'])})"
            cells.append(v)
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {int(round(s['n_defect']))} |")
    if pairs:
        lines += ["", "| 비교(A 대 B) | A | B | 차이 | 95% 구간 | McNemar | Holm 보정 | 장수 |", "|---" * 8 + "|"]
        for name, p in pairs.items():
            lines.append(f"| {name} | {pct(p['acc_a'])} | {pct(p['acc_b'])} | {100 * p['diff']:+.1f} "
                         f"| [{100 * p['ci95'][0]:+.1f}, {100 * p['ci95'][1]:+.1f}] | {format_p(p['p'])} "
                         f"| {format_p(p['p_holm'])} | {p['n']} |")
    if pairs_scope:
        lines += ["", "| 비교(A 대 B) | 범위 | A | B | 차이 | 95% 구간 | McNemar | 장수 |", "|---" * 8 + "|"]
        for name, by_scope in pairs_scope.items():
            for sc, p in by_scope.items():
                lines.append(f"| {name} | {SCOPE_LABEL[sc]} | {pct(p['acc_a'])} | {pct(p['acc_b'])} | {100 * p['diff']:+.1f} "
                             f"| [{100 * p['ci95'][0]:+.1f}, {100 * p['ci95'][1]:+.1f}] | {format_p(p['p'])} | {p['n']} |")
    return "\n".join(lines) + "\n"


def per_product_markdown(summary: dict) -> str:
    cats = sorted({c for s in summary.values() for c in s.get("per_product", {})})
    names = list(summary)
    lines = ["| 제품 | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for c in cats:
        lines.append(f"| {c} | " + " | ".join(pct(summary[n].get("per_product", {}).get(c)) for n in names) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlm", action="append", default=[], metavar="NAME=GLOB", help="VLM 예측 파일 패턴")
    ap.add_argument("--pair", action="append", default=[], metavar="A:B", help="짝지어 비교할 두 방법")
    ap.add_argument("--pair-scope", action="append", default=[], metavar="A:B",
                    help="공통/고유 유형과 정상 판정률로 나누어 짝지어 비교할 두 방법")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out-dir", type=Path, default=TABLE_DIR)
    args = ap.parse_args()

    scope = type_scope(load_lexicon())
    methods = baseline_methods()
    for spec in args.vlm:
        name, pattern = spec.split("=", 1)
        methods[name] = load_glob(pattern)
    summary = {name: evaluate(rows, scope) for name, rows in methods.items()}
    summary.update(kshot_summary(scope))

    pairs = {}
    for spec in args.pair:
        a, b = spec.split(":")
        pairs[f"{a} 대 {b}"] = pairwise(methods[a], methods[b], scope, args.n_boot)
    if pairs:
        adj = holm({k: v["p"] for k, v in pairs.items()})
        for k in pairs:
            pairs[k]["p_holm"] = adj[k]

    pairs_scope = {}
    for spec in args.pair_scope:
        a, b = spec.split(":")
        pairs_scope[f"{a} 대 {b}"] = pairwise_scope(methods[a], methods[b], scope, args.n_boot)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "pairs": pairs, "pairs_scope": pairs_scope}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    md = summary_markdown(summary, pairs, pairs_scope)
    (args.out_dir / "summary.md").write_text(md, encoding="utf-8")
    (args.out_dir / "per_product.md").write_text(per_product_markdown(summary), encoding="utf-8")
    print(md)
    print(f"저장: {args.out_dir / 'summary.json'}, summary.md, per_product.md")


if __name__ == "__main__":
    main()
