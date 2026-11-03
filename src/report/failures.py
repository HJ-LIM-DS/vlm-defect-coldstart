"""VLM 예측의 실패 사례를 정리한다(reports/tables/failures.md).

무엇을 보는가
- 혼동 쌍: 제품별로 정답 유형과 예측 유형의 쌍을 세어 가장 자주 틀린 방향을 뽑는다. 사전의 한국어 이름으로 적어
  어떤 정의가 서로 섞이는지 읽을 수 있게 한다.
- 결함을 정상으로 답한 비율(놓침)과 정상을 결함으로 답한 비율(오경보)을 제품별로 센다.
- 비교 방법(기본은 같은 입력의 zero-shot)과 견주어 적응 뒤 고쳐진 이미지와 새로 틀린 이미지 수를 제품별로 센다.
- 자주 틀린 혼동 쌍마다 모델이 쓴 근거 문장 몇 개를 그대로 둔다. 이미지 없이도 모델이 무엇을 보았다고 말하는지 읽기 위해서다.

사용법:
    python -m src.report.failures --main l4b="runs/l4b/f*/eval_predictions.jsonl" --ref z4b_noref="runs/z4b_noref/f*_predictions.jsonl"
"""

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.data.lexicon import load_lexicon, type_scope


def load_glob(pattern: str) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(pattern)):
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def name_of(lex: dict, cat: str, key) -> str:
    if key == "good":
        return "정상"
    if key is None:
        return "형식 위반"
    d = lex["products"][cat]["defects"].get(key)
    return d["name_ko"] if d else str(key)


def confusions(rows: list[dict], lex: dict, scope: dict, top: int) -> list[dict]:
    pairs = Counter()
    total = Counter()
    for r in rows:
        if r["label"] != "defect":
            continue
        total[(r["category"], r["defect_type"])] += 1
        if r["pred_key"] != r["defect_type"]:
            pairs[(r["category"], r["defect_type"], r["pred_key"])] += 1
    out = []
    for (cat, truth, pred), n in pairs.most_common():
        out.append({"category": cat, "truth": truth, "pred": pred, "n": n, "n_truth": total[(cat, truth)],
                    "truth_name": name_of(lex, cat, truth), "pred_name": name_of(lex, cat, pred),
                    "truth_scope": scope.get((cat, truth), ""),
                    "pred_scope": scope.get((cat, pred), "정상" if pred == "good" else "")})
    return out[:top]


def miss_and_alarm(rows: list[dict]) -> dict[str, dict]:
    out = defaultdict(lambda: {"defect": 0, "miss": 0, "good": 0, "alarm": 0, "format_bad": 0})
    for r in rows:
        s = out[r["category"]]
        if not r["format_ok"]:
            s["format_bad"] += 1
        if r["label"] == "defect":
            s["defect"] += 1
            s["miss"] += r["pred_key"] == "good"
        else:
            s["good"] += 1
            s["alarm"] += r["pred_key"] != "good"
    return dict(sorted(out.items()))


def fixed_and_broken(main: list[dict], ref: list[dict]) -> dict[str, dict]:
    by_id = {r["image_id"]: r for r in ref}
    out = defaultdict(lambda: {"n": 0, "fixed": 0, "broken": 0, "both_wrong": 0})
    for r in main:
        if r["label"] != "defect" or r["image_id"] not in by_id:
            continue
        a = r["pred_key"] == r["defect_type"]
        b = by_id[r["image_id"]]["pred_key"] == r["defect_type"]
        s = out[r["category"]]
        s["n"] += 1
        s["fixed"] += a and not b
        s["broken"] += b and not a
        s["both_wrong"] += (not a) and (not b)
    return dict(sorted(out.items()))


def definition_copy_rate(rows: list[dict], lex: dict) -> dict:
    """근거 문장이 예측한 유형의 사전 정의와 글자 그대로 같은 비율. 적응 모델이 이미지를 서술하는 대신 정의를 되뇌는지 본다."""
    n = same = 0
    distinct = set()
    for r in rows:
        if not r.get("reason") or r["pred_key"] in (None, "good"):
            continue
        d = lex["products"][r["category"]]["defects"].get(r["pred_key"])
        if d is None:
            continue
        n += 1
        same += r["reason"].strip().rstrip(".") == d["definition"].strip().rstrip(".")
        distinct.add(r["reason"].strip())
    return {"n": n, "copied": same, "rate": same / n if n else float("nan"), "distinct": len(distinct)}


def reasons_for(rows: list[dict], cat: str, truth: str, pred, k: int) -> list[str]:
    seen, out = set(), []
    for r in rows:
        if r["category"] == cat and r["defect_type"] == truth and r["pred_key"] == pred and r.get("reason"):
            if r["reason"] not in seen:
                seen.add(r["reason"])
                out.append(f"{r['image_id'].split('/')[-1]}: {r['reason']}")
            if len(out) >= k:
                break
    return out


def markdown(main_name: str, ref_name: str, conf: list[dict], ma: dict, fb: dict, reasons: dict, copy: dict) -> str:
    L = [f"# 실패 사례: {main_name}", "", f"비교 대상은 {ref_name}(같은 입력)이다. 혼동 쌍은 결함 이미지에서 정답과 다른 예측을 (제품, 정답, 예측)으로 세어 많은 순으로 적었다.",
         "", "## 가장 자주 틀린 방향", "",
         "| 제품 | 정답 | 범위 | 예측 | 범위 | 장수 / 정답 장수 |", "|---|---|---|---|---|---|"]
    for c in conf:
        L.append(f"| {c['category']} | {c['truth_name']} ({c['truth']}) | {c['truth_scope']} | {c['pred_name']} | {c['pred_scope']} | {c['n']} / {c['n_truth']} |")
    L += ["", "## 놓침과 오경보", "", "결함을 정상이라 답한 장수(놓침)와 정상을 결함이라 답한 장수(오경보)를 제품별로 센다.", "",
          "| 제품 | 결함 장수 | 놓침 | 정상 장수 | 오경보 | 형식 위반 |", "|---|---|---|---|---|---|"]
    for cat, s in ma.items():
        L.append(f"| {cat} | {s['defect']} | {s['miss']} | {s['good']} | {s['alarm']} | {s['format_bad']} |")
    tot = Counter()
    for s in ma.values():
        tot.update(s)
    L.append(f"| 합계 | {tot['defect']} | {tot['miss']} | {tot['good']} | {tot['alarm']} | {tot['format_bad']} |")
    L += ["", f"## {ref_name}와 견준 변화", "", "같은 결함 이미지에서 적응 뒤 맞게 바뀐 장수(고침)와 틀리게 바뀐 장수(망침)다.", "",
          "| 제품 | 결함 장수 | 고침 | 망침 | 둘 다 틀림 |", "|---|---|---|---|---|"]
    for cat, s in fb.items():
        L.append(f"| {cat} | {s['n']} | {s['fixed']} | {s['broken']} | {s['both_wrong']} |")
    tot = Counter()
    for s in fb.values():
        tot.update(s)
    L.append(f"| 합계 | {tot['n']} | {tot['fixed']} | {tot['broken']} | {tot['both_wrong']} |")
    L += ["", "## 근거 문장의 사전 복사율", "",
          "결함이라 답한 이미지에서 근거 문장이 예측한 유형의 사전 정의와 글자 그대로 같은 비율이다. 서로 다른 근거 문장의 수를 함께 적는다.", "",
          "| 방법 | 결함 답변 장수 | 정의와 같은 문장 | 비율 | 서로 다른 문장 수 |", "|---|---|---|---|---|"]
    for name, c in copy.items():
        L.append(f"| {name} | {c['n']} | {c['copied']} | {100 * c['rate']:.1f}% | {c['distinct']} |")
    L += ["", "## 자주 틀린 혼동 쌍의 근거 문장", "", "모델이 쓴 근거를 그대로 옮겼다(이미지 번호: 근거). 같은 문장은 한 번만 적는다.", ""]
    for key, lines in reasons.items():
        L.append(f"- {key}")
        L += [f"  - {x}" for x in lines]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", required=True, metavar="NAME=GLOB")
    ap.add_argument("--ref", required=True, metavar="NAME=GLOB")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--reasons", type=int, default=3, help="혼동 쌍마다 남길 근거 문장 수")
    ap.add_argument("--out", type=Path, default=Path("reports/tables/failures.md"))
    args = ap.parse_args()

    lex = load_lexicon()
    scope = type_scope(lex)
    main_name, main_pat = args.main.split("=", 1)
    ref_name, ref_pat = args.ref.split("=", 1)
    main_rows, ref_rows = load_glob(main_pat), load_glob(ref_pat)
    conf = confusions(main_rows, lex, scope, args.top)
    reasons = {f"{c['category']}, 정답 {c['truth_name']}, 예측 {c['pred_name']}": reasons_for(main_rows, c["category"], c["truth"], c["pred"], args.reasons)
               for c in conf[:8]}
    copy = {main_name: definition_copy_rate(main_rows, lex), ref_name: definition_copy_rate(ref_rows, lex)}
    md = markdown(main_name, ref_name, conf, miss_and_alarm(main_rows), fixed_and_broken(main_rows, ref_rows), reasons, copy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
