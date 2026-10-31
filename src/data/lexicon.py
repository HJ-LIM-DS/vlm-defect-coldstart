"""결함 사전(lexicon/mvtec_ad.yaml) 로더와 공통/고유 유형 계산.

- 공통 유형: 같은 키(폴더 이름)가 두 제품 이상에 있는 결함 유형. 다른 제품에서 배운 어휘가 전이될 수 있다.
- 고유 유형: 한 제품에만 있는 키. 정의 문장을 읽어야만 맞힐 수 있다고 가정한다.
이 구분은 키 일치로만 계산한다. 뜻이 비슷한 이름(예: scratch와 scratch_head)을 사람이 묶으면 결과를 보고 고를 여지가 생기기 때문이다.

사용법:
    python -m src.data.lexicon --check data/index.csv --coverage data/lexicon_coverage.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import yaml

DEFAULT_PATH = Path("lexicon/mvtec_ad.yaml")


def load_lexicon(path: Path = DEFAULT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        lex = yaml.safe_load(f)
    validate(lex)
    return lex


def validate(lex: dict) -> None:
    for cat, p in lex["products"].items():
        for field in ("name_ko", "name_en", "kind", "normal", "normal_en", "defects"):
            if field not in p:
                raise ValueError(f"{cat}: '{field}' 누락")
        names = [d["name_ko"] for d in p["defects"].values()]
        if len(names) != len(set(names)):
            raise ValueError(f"{cat}: name_ko 중복 {names}")
        for key, d in p["defects"].items():
            for field in ("name_ko", "definition", "definition_en"):
                if not d.get(field):
                    raise ValueError(f"{cat}/{key}: '{field}' 누락")


def defect_keys(lex: dict) -> dict[str, list[str]]:
    return {cat: sorted(p["defects"]) for cat, p in lex["products"].items()}


def key_products(lex: dict) -> dict[str, list[str]]:
    """결함 키별로 그 키를 가진 제품 목록."""
    out = defaultdict(list)
    for cat, keys in defect_keys(lex).items():
        for k in keys:
            out[k].append(cat)
    return {k: sorted(v) for k, v in sorted(out.items())}


def type_scope(lex: dict) -> dict[tuple[str, str], str]:
    """(제품, 키) -> 'shared' | 'unique'."""
    kp = key_products(lex)
    return {(cat, k): ("shared" if len(kp[k]) >= 2 else "unique")
            for cat, keys in defect_keys(lex).items() for k in keys}


def name_to_key(lex: dict, category: str) -> dict[str, str]:
    """출력 문장의 한국어 표시 이름을 사전 키로 되돌리는 표."""
    return {d["name_ko"]: k for k, d in lex["products"][category]["defects"].items()}


def check_against_index(lex: dict, index_path: Path) -> list[str]:
    """사전의 (제품, 키) 집합과 인덱스의 결함 폴더 집합이 같은지 확인하고 불일치를 돌려준다."""
    seen = defaultdict(set)
    with open(index_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["defect_type"]:
                seen[r["category"]].add(r["defect_type"])
    problems = []
    for cat, keys in defect_keys(lex).items():
        idx = seen.get(cat, set())
        for k in sorted(set(keys) - idx):
            problems.append(f"{cat}/{k}: 사전에만 있음")
        for k in sorted(idx - set(keys)):
            problems.append(f"{cat}/{k}: 인덱스에만 있음")
    for cat in sorted(set(seen) - set(lex["products"])):
        problems.append(f"{cat}: 인덱스에만 있는 제품")
    return problems


def coverage_rows(lex: dict, index_path: Path | None = None) -> list[dict]:
    """(제품, 키)별 공통/고유 구분과, 인덱스가 있으면 테스트 결함 장수."""
    counts = defaultdict(int)
    if index_path is not None:
        with open(index_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["defect_type"]:
                    counts[(r["category"], r["defect_type"])] += 1
    kp = key_products(lex)
    rows = []
    for cat, keys in defect_keys(lex).items():
        for k in keys:
            rows.append({
                "category": cat, "defect_type": k, "name_ko": lex["products"][cat]["defects"][k]["name_ko"],
                "scope": "shared" if len(kp[k]) >= 2 else "unique",
                "n_products_with_key": len(kp[k]),
                "n_test_images": counts.get((cat, k), ""),
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--check", type=Path, help="data/index.csv 와 대조")
    ap.add_argument("--coverage", type=Path, help="공통/고유 표를 CSV로 저장")
    args = ap.parse_args()

    lex = load_lexicon(args.lexicon)
    scope = type_scope(lex)
    n_shared = sum(v == "shared" for v in scope.values())
    print(f"제품 {len(lex['products'])}개, 결함 유형 {len(scope)}개 (공통 {n_shared}, 고유 {len(scope) - n_shared})")
    for k, cats in key_products(lex).items():
        if len(cats) >= 2:
            print(f"  공통 {k:<22} {', '.join(cats)}")
    if args.check:
        problems = check_against_index(lex, args.check)
        print("인덱스 대조: " + ("일치" if not problems else f"불일치 {len(problems)}건"))
        for p in problems:
            print("  " + p)
    if args.coverage:
        rows = coverage_rows(lex, args.check)
        with open(args.coverage, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"공통/고유 표 저장: {args.coverage}")


if __name__ == "__main__":
    main()
