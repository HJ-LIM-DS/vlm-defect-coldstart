# 실패 사례: l4b

비교 대상은 z4b_noref(같은 입력)이다. 혼동 쌍은 결함 이미지에서 정답과 다른 예측을 (제품, 정답, 예측)으로 세어 많은 순으로 적었다.

## 가장 자주 틀린 방향

| 제품 | 정답 | 범위 | 예측 | 범위 | 장수 / 정답 장수 |
|---|---|---|---|---|---|
| metal_nut | 뒤집힘 (flip) | unique | 정상 | 정상 | 19 / 23 |
| zipper | 안쪽 천 손상 (fabric_interior) | unique | 복합 결함 | shared | 14 / 16 |
| hazelnut | 베임 (cut) | shared | 균열 | shared | 13 / 17 |
| bottle | 작은 파손 (broken_small) | unique | 이물 혼입 | shared | 12 / 22 |
| tile | 회색 줄 (gray_stroke) | unique | 기름 얼룩 | unique | 12 / 16 |
| zipper | 이빨 거침 (rough) | shared | 이빨 파손 | unique | 12 / 17 |
| grid | 휨 (bent) | shared | 끊어짐 | unique | 11 / 12 |
| cable | 케이블 자리 바뀜 (cable_swap) | unique | 정상 | 정상 | 10 / 12 |
| carpet | 베임 (cut) | shared | 실 풀림 | shared | 9 / 17 |
| carpet | 구멍 (hole) | shared | 실 풀림 | shared | 9 / 17 |
| capsule | 찔림 (poke) | shared | 인쇄 불량 | shared | 9 / 21 |
| metal_nut | 휨 (bent) | shared | 변색 | shared | 9 / 25 |
| transistor | 다리 절단 (cut_lead) | unique | 다리 휨 | unique | 9 / 10 |
| zipper | 이빨 벌어짐 (split_teeth) | unique | 이빨 눌림 | unique | 9 / 18 |
| metal_nut | 휨 (bent) | shared | 긁힘 | shared | 8 / 25 |
| screw | 측면 나사산 손상 (thread_side) | unique | 정상 | 정상 | 8 / 23 |
| screw | 측면 나사산 손상 (thread_side) | unique | 상단 나사산 손상 | unique | 8 / 23 |
| transistor | 위치 이탈 (misplaced) | unique | 다리 휨 | unique | 8 / 10 |
| zipper | 이빨 눌림 (squeezed_teeth) | unique | 정상 | 정상 | 8 / 16 |
| bottle | 큰 파손 (broken_large) | unique | 이물 혼입 | shared | 7 / 20 |

## 놓침과 오경보

결함을 정상이라 답한 장수(놓침)와 정상을 결함이라 답한 장수(오경보)를 제품별로 센다.

| 제품 | 결함 장수 | 놓침 | 정상 장수 | 오경보 | 형식 위반 |
|---|---|---|---|---|---|
| bottle | 63 | 5 | 20 | 2 | 0 |
| cable | 92 | 28 | 58 | 0 | 0 |
| capsule | 109 | 9 | 23 | 1 | 0 |
| carpet | 89 | 1 | 28 | 1 | 0 |
| grid | 57 | 1 | 21 | 1 | 0 |
| hazelnut | 70 | 1 | 40 | 24 | 0 |
| leather | 92 | 0 | 32 | 2 | 0 |
| metal_nut | 93 | 20 | 22 | 16 | 0 |
| pill | 141 | 10 | 26 | 5 | 0 |
| screw | 119 | 20 | 41 | 4 | 0 |
| tile | 84 | 1 | 33 | 5 | 0 |
| toothbrush | 30 | 0 | 12 | 2 | 0 |
| transistor | 40 | 2 | 60 | 60 | 0 |
| wood | 60 | 1 | 19 | 4 | 0 |
| zipper | 119 | 19 | 32 | 1 | 0 |
| 합계 | 1258 | 118 | 467 | 128 | 0 |

## z4b_noref와 견준 변화

같은 결함 이미지에서 적응 뒤 맞게 바뀐 장수(고침)와 틀리게 바뀐 장수(망침)다.

| 제품 | 결함 장수 | 고침 | 망침 | 둘 다 틀림 |
|---|---|---|---|---|
| bottle | 63 | 16 | 2 | 26 |
| cable | 92 | 19 | 7 | 51 |
| capsule | 109 | 13 | 3 | 31 |
| carpet | 89 | 6 | 6 | 23 |
| grid | 57 | 17 | 7 | 20 |
| hazelnut | 70 | 5 | 8 | 20 |
| leather | 92 | 11 | 4 | 15 |
| metal_nut | 93 | 10 | 6 | 40 |
| pill | 141 | 33 | 2 | 64 |
| screw | 119 | 21 | 14 | 38 |
| tile | 84 | 9 | 4 | 28 |
| toothbrush | 30 | 0 | 0 | 0 |
| transistor | 40 | 1 | 4 | 17 |
| wood | 60 | 8 | 5 | 11 |
| zipper | 119 | 13 | 23 | 56 |
| 합계 | 1258 | 182 | 95 | 440 |

## 근거 문장의 사전 복사율

결함이라 답한 이미지에서 근거 문장이 예측한 유형의 사전 정의와 글자 그대로 같은 비율이다. 서로 다른 근거 문장의 수를 함께 적는다.

| 방법 | 결함 답변 장수 | 정의와 같은 문장 | 비율 | 서로 다른 문장 수 |
|---|---|---|---|---|
| l4b | 1268 | 1268 | 100.0% | 64 |
| z4b_noref | 1557 | 0 | 0.0% | 1153 |

## 자주 틀린 혼동 쌍의 근거 문장

모델이 쓴 근거를 그대로 옮겼다(이미지 번호: 근거). 같은 문장은 한 번만 적는다.

- metal_nut, 정답 뒤집힘, 예측 정상
- zipper, 정답 안쪽 천 손상, 예측 복합 결함
  - 000: 두 가지 이상의 결함이 한 이미지에 함께 있다.
- hazelnut, 정답 베임, 예측 균열
  - 001: 껍질에 금이 가거나 갈라졌다.
- bottle, 정답 작은 파손, 예측 이물 혼입
  - 000: 병 안이나 입구에 제품이 아닌 이물질이 들어 있다.
- tile, 정답 회색 줄, 예측 기름 얼룩
  - 000: 표면에 기름이 묻어 번들거리거나 어두운 얼룩이 있다.
- zipper, 정답 이빨 거침, 예측 이빨 파손
  - 000: 지퍼 이빨이 부러지거나 빠졌다.
- grid, 정답 휨, 예측 끊어짐
  - 001: 격자 선이 끊어져 틈이 생겼다.
- cable, 정답 케이블 자리 바뀜, 예측 정상
