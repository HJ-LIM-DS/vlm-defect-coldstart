"""VLM 프롬프트 템플릿과 출력 파서. 템플릿은 첫 모델을 돌리기 전에 고정하고 그 뒤 바꾸지 않는다.

메시지 구성(이미지는 있는 것만 넣는다):
    [검사 이미지] [이상 영역 crop] [정상 참조]
    제품과 정상 규격, 결함 유형 목록(이름과 정의, 순서는 이미지마다 무작위), 답 형식.

출력 형식:
    유형: <목록의 이름 하나, 또는 정상>
    근거: <한 문장>
"""

import random
import re

from src.data.lexicon import name_to_key

NORMAL = "정상"

INSTRUCTION = (
    "첫 번째 이미지는 검사할 제품 사진이다."
    "{crop_note}{ref_note}\n\n"
    "제품: {product}\n"
    "정상 규격: {normal}\n"
    "이 제품에서 생길 수 있는 결함 유형:\n{defect_list}\n\n"
    "검사 이미지에 결함이 없으면 '유형: 정상' 한 줄만 쓴다. "
    "결함이 있으면 위 목록에서 가장 맞는 유형 하나를 골라 아래 형식으로 두 줄만 쓴다.\n"
    "유형: <유형 이름>\n"
    "근거: <무엇이 어떻게 잘못됐는지 한 문장>"
)
CROP_NOTE = " 두 번째 이미지는 검사 이미지에서 이상 점수가 가장 높은 영역을 확대한 것이다."
REF_NOTE = " 마지막 이미지는 같은 제품의 정상 사진이다."

PROBE_INSTRUCTION = "이 제품 사진에 결함이 있다면 결함의 종류를 영어 한 단어나 짧은 구로만 답하라."


def defect_list_text(lex: dict, category: str, with_definitions: bool, rng: random.Random) -> str:
    items = list(lex["products"][category]["defects"].items())
    rng.shuffle(items)
    if with_definitions:
        return "\n".join(f"- {d['name_ko']}: {d['definition']}" for _, d in items)
    return "\n".join(f"- {d['name_ko']}" for _, d in items)


def build_messages(lex: dict, category: str, image, crop=None, ref=None, with_definitions: bool = True,
                   seed: int = 0) -> list[dict]:
    """Qwen3-VL chat template에 넣을 messages. 이미지 인자는 PIL.Image 또는 경로."""
    p = lex["products"][category]
    rng = random.Random(seed)
    content = [{"type": "image", "image": image}]
    if crop is not None:
        content.append({"type": "image", "image": crop})
    if ref is not None:
        content.append({"type": "image", "image": ref})
    text = INSTRUCTION.format(
        crop_note=CROP_NOTE if crop is not None else "",
        ref_note=REF_NOTE if ref is not None else "",
        product=p["name_ko"], normal=p["normal"],
        defect_list=defect_list_text(lex, category, with_definitions, rng),
    )
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def build_probe_messages(image) -> list[dict]:
    """사전학습 노출 탐침: 사전 없이 결함 종류를 영어로 묻는다."""
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROBE_INSTRUCTION}]}]


def target_text(lex: dict, category: str, defect_key: str | None, reason: str | None = None) -> str:
    """학습 정답 문자열. 정상이면 한 줄, 결함이면 두 줄."""
    if defect_key is None:
        return f"유형: {NORMAL}"
    d = lex["products"][category]["defects"][defect_key]
    return f"유형: {d['name_ko']}\n근거: {reason or d['definition']}"


_TYPE_RE = re.compile(r"유형\s*[:：]\s*(.+)")
_REASON_RE = re.compile(r"근거\s*[:：]\s*(.+)")


def parse_output(text: str, lex: dict, category: str) -> dict:
    """모델 출력을 {key, name, reason, format_ok}로 바꾼다.

    key는 사전 키, 정상이면 'good', 목록에 없는 이름이면 None(형식 위반). 이름은 정확히 일치해야 한다.
    """
    m = _TYPE_RE.search(text)
    if not m:
        return {"key": None, "name": None, "reason": None, "format_ok": False}
    name = m.group(1).strip().rstrip(".。")
    r = _REASON_RE.search(text)
    reason = r.group(1).strip() if r else None
    if name == NORMAL:
        return {"key": "good", "name": name, "reason": reason, "format_ok": True}
    key = name_to_key(lex, category).get(name)
    return {"key": key, "name": name, "reason": reason, "format_ok": key is not None and reason is not None}
