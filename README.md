# 2026-2-HAI-ProjectTeam7

## Deepfake Detection: Generalization to Unseen Manipulations

**학습 데이터의 다양성이 딥페이크 탐지 모델의 일반화 성능에 미치는 영향 분석**

본 프로젝트는 DF40 데이터셋을 활용하여 딥페이크 영상 탐지 모델을 구현하고, **학습 과정에서 접하지 못한 조작 기법(Unseen Manipulation)**에 대한 일반화 성능을 분석하는 것을 목표로 한다.

사전학습된 CNN 및 ViT 모델을 Fine-tuning하고, 학습 데이터의 조작 기법 다양성(Training Diversity)에 따른 탐지 성능 차이를 비교한다.

## 1. 문제 정의 및 연구 목표

딥페이크 생성 기술이 발전하면서 다양한 조작 기법이 등장하고 있다. 기존 탐지 모델은 학습에 사용된 조작 기법에서는 높은 정확도를 보이더라도, 새로운 조작 기법에 대해서는 성능이 저하될 수 있다.

이에 다음 연구 질문을 중심으로 프로젝트를 진행한다.

- **RQ1 (핵심):** 학습 데이터의 조작 기법 다양성이 Unseen Manipulation 탐지 성능에 어떤 영향을 미치는가?
- **RQ2 (확장):** CNN과 ViT 기반 모델은 일반화 성능에서 어떤 차이를 보이는가?
- **RQ3 (선택):** 압축 및 해상도 변화가 탐지 성능에 미치는 영향은 무엇인가?

## 2. 데이터셋 — DF40

NeurIPS 2024에 발표된 **DF40** 데이터셋을 사용한다.

DF40은 Face Swapping(10), Face Reenactment(13), Entire Face Synthesis(12), Face Editing(5)의 총 40개 조작 기법을 포함한다.

초기에는 영상 단위 탐지를 목표로 **Face Swapping(FS)** 유형의 일부 조작 기법을 사용한다.

| 구분 | 조작 기법 후보 |
|---|---|
| Training | SimSwap, FaceSwap, FSGAN |
| Unseen Test | BlendFace |

*실제 사용할 subset과 데이터 규모는 접근성 및 파일 구조 확인 후 확정한다.*

## 3. 모델 및 처리 파이프라인

**Video → Frame Extraction → Face Detection/Cropping → Pretrained Model → Frame-level Prediction → Video-level Prediction**

초기에는 DF40에서 제공하는 얼굴 crop 이미지를 활용하여 Frame-level 분류 모델을 학습하고, 이후 프레임별 Fake 확률을 집계해 Video-level Detection으로 확장한다.

| 모델 | 역할 |
|---|---|
| EfficientNet-B0 | CNN Baseline |
| ViT-B/16 | Transformer 비교 모델 (확장) |
| Xception | 딥페이크 탐지 선행연구 참고 |

사전학습 모델의 분류층을 이진 분류에 맞게 수정하여 Fine-tuning을 진행할 계획이다.

## 4. 실험 설계

### Experiment 1. CNN Baseline

- 특정 조작 기법의 Real/Fake 데이터로 CNN Fine-tuning
- Seen / Unseen Manipulation에 대한 성능 비교
- ROC-AUC, F1-score, Accuracy 측정

### Experiment 2. Training Diversity (핵심 실험)

학습 데이터의 전체 샘플 수를 통제한 상태에서 조작 기법의 다양성을 변경한다.

- **Case A:** 단일 조작 기법으로 학습
- **Case B:** 복수 조작 기법으로 학습
- 동일한 Unseen Test Set에서 일반화 성능 비교

### Experiment 3. CNN vs ViT (확장)

동일한 데이터 분할 및 평가 조건에서 CNN과 ViT의 Seen/Unseen 성능을 비교한다.

### Experiment 4. Video-level Detection (확장)

프레임별 Fake 확률을 평균하여 영상 단위 판별 결과를 생성하고, 프레임별 의심도를 시각화한다.

**실험 규칙**
- 원본 영상 및 파생 데이터 간 Data Leakage 방지
- Train / Validation / Test 분리
- Test Set을 모델 튜닝에 사용하지 않음
- 동일 조건 비교를 위한 Seed 및 학습 설정 기록

## 5. 개발 현황

현재는 **프로젝트 기획 및 실험 설계 단계**이며, 실제 모델 학습과 성능 평가는 진행 전이다.

| 항목 | 상태 |
|---|---|
| 프로젝트 주제 및 연구 방향 설정 | 완료 |
| DF40 데이터셋 선정 | 완료 |
| CNN / ViT 기반 모델 구조 논의 | 완료 |
| Seen / Unseen 실험 방향 설계 | 초안 작성 |
| 데이터 다운로드 및 전처리 | 예정 |
| CNN Baseline Fine-tuning | 예정 |
| 일반화 성능 평가 | 예정 |
| ViT 비교 및 Video-level 확장 | 예정 |

## 6. 개발 계획

| 단계 | 목표 | 산출물 |
|---|---|---|
| Step 1 | DF40 Subset 확보 및 데이터 검증 | 데이터 로딩 파이프라인 |
| Step 2 | CNN Baseline Fine-tuning | 학습 코드 및 Loss 그래프 |
| Step 3 | Seen / Unseen 성능 평가 | ROC-AUC, F1 결과표 |
| Step 4 | Training Diversity 비교 | 일반화 성능 비교 그래프 |
| Step 5 | CNN / ViT 비교 | 모델별 성능 분석 |
| Step 6 | Video-level Detection | 프레임별 확률 시각화 및 데모 |

**Step 1~4를 핵심 개발 범위**로 설정하며, 진행 상황에 따라 Step 5~6으로 확장한다.

## 7. 예상 결과물

- CNN 기반 딥페이크 탐지 모델
- Seen / Unseen Manipulation 성능 비교
- Training Diversity에 따른 일반화 성능 분석
- CNN / ViT 비교 실험 (확장)
- 영상 단위 탐지 및 프레임별 확률 시각화 (확장)

## References

- [DF40: Toward Next-Generation Deepfake Detection (NeurIPS 2024)](https://github.com/YZY-stack/DF40)
- [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)

---

*2026.10.29 중간발표 기준 프로젝트 계획. 실제 구현 및 실험 진행에 따라 업데이트 예정.*
