# 결함 사전만 주고 새 제품을 검사하기: LoRA 적응 VLM의 콜드 스타트 실험

처음 보는 제품의 결함 라벨이 한 장도 없을 때, 결함 유형의 이름과 한 줄 정의만 글로 주면 비전 언어 모델(VLM)이 결함 유형을 맞힐 수 있는가. 그리고 전문 few-shot 분류기가 그 성능을 따라잡으려면 새 제품의 라벨이 몇 장 필요한가. MVTec AD 15개 제품을 제품 단위 5-fold로 나누어, 12개 제품으로 QLoRA 적응한 Qwen3-VL이 나머지 3개 제품을 사전만 보고 검사하게 했다. 로컬 GPU 1장(RTX 3060 12GB)에서 학습하고 평가한다.

```
유형: 균열
근거: 캡슐 껍질에 금이 간 선이 보인다
```

> 상태(2026-11-01): 준비 단계. 데이터 인덱스, 결함 사전, 제품 분할, 연구 설계 문서를 만드는 중이다. 결과는 아직 없다.

## 무엇을 재는가

| 질문 | 비교 |
|---|---|
| 라벨 0장에서 VLM이 비-VLM보다 나은가 | LoRA VLM 대 교차 제품 분류기, CLIP 텍스트 매칭 |
| 라벨이 몇 장이면 분류기가 따라잡는가 | 유형당 k장(1, 2, 5, 10) few-shot 분류기와의 교차점 |
| VLM은 정의 문장을 읽는가 | 정의를 뺀 사전으로 다시 평가, 공통 유형과 고유 유형을 나누어 비교 |
| 라벨이 충분하면 어느 쪽이 나은가 | 전체 라벨 분류기(VLM에 불리한 대조군) |

가설과 판정 기준은 결과를 보기 전에 [`docs/preregistration.md`](docs/preregistration.md)에 적었다.

## 저장소 구조

```
lexicon/        결함 사전 (제품별 정상 규격, 결함 유형 이름과 정의)
data/           인덱스와 분할 (이미지는 포함하지 않음)
src/data/       인덱스, 사전 로더, 제품 분할
src/detector/   PatchCore 이상 맵과 crop
src/baselines/  교차 제품 분류기, CLIP 매칭, few-shot 분류기
src/vlm/        프롬프트, 학습 표본, LoRA 학습, 추론, 출력 파서
src/eval/       지표와 통계
src/report/     표와 그림 생성
ops/            학습 대기열과 감독자
tests/
reports/        표, 그림, 예측 원본
```

## 데이터

MVTec AD는 [공식 페이지](https://www.mvtec.com/company/research/datasets/mvtec-ad)에서 받는다(CC BY-NC-SA 4.0, 재배포하지 않음). 아카이브를 `data/mvtec_ad/`에 풀고 인덱스를 만든다.

```
python -m src.data.index_mvtec --root data/mvtec_ad --out data/index.csv
python -m src.data.lexicon --check data/index.csv --coverage data/lexicon_coverage.csv
python -m src.data.splits --out data/splits.csv
```

## 환경

Windows 11, Python 3.11, CUDA 12.4. `requirements.txt`의 버전을 그대로 쓴다.

```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 인용

Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., Steger, C. The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. IJCV 129(4), 2021.
