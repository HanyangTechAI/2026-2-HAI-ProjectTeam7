# 2026-2-HAI-ProjectTeam7

## Deepfake Detection: Training Diversity & Unseen Generalization

**제한된 학습 데이터에서 조작 기법의 다양성이 딥페이크 탐지 모델의 일반화 성능에 미치는 영향 분석**

본 프로젝트는 DF40 데이터셋을 활용해 사전학습된 CNN 기반 Real/Fake 분류 모델을 Fine-tuning하고, 학습 데이터의 조작 기법 구성에 따라 미학습 기법(Unseen Manipulation) 탐지 성능이 어떻게 달라지는지 분석한다.

특히 **고정된 학습 데이터 예산**에서 단일 기법 학습과 다중 기법 학습을 비교하고, 일반화 성능의 변화와 기존 기법(Seen)에 대한 성능 저하 가능성을 함께 평가한다.

> **현재 상태 (2026-10-29): 기획 및 실험 설계 단계.** 실제 DF40 데이터 검증, 모델 학습, 성능 측정은 아직 진행하지 않았다.

## 1. Motivation & Research Questions

특정 딥페이크 생성 기법으로 학습한 탐지기는 학습에 사용하지 않은 생성 기법에서 성능이 저하될 수 있다. 모든 새로운 기법의 데이터를 사전에 확보하기 어렵기 때문에, **같은 양의 데이터로 학습할 때 어떤 구성 전략이 새로운 조작 기법에 유리한가**를 알아볼 필요가 있다.

- **RQ1 (핵심):** 전체 학습 데이터 수가 같을 때 단일 기법 학습과 다중 기법 학습은 Unseen 탐지 성능에서 어떤 차이를 보이는가?
- **RQ2 (핵심):** 다중 기법 학습은 공통 Seen 기법의 성능을 희생하면서 Unseen 성능을 개선하는가? (Trade-off)
- **RQ3 (확장):** CNN과 ViT를 사용하면 위 결과가 달라지는가?

본 프로젝트는 새로운 SOTA 탐지기를 제안하기보다는 **DF40 Face Swapping 환경에서 데이터 구성 전략을 재현 가능하게 비교·분석하는 Data-centric 실험**을 목표로 한다.

## 2. Dataset — DF40

DF40 (NeurIPS 2024)은 총 40개의 조작 기법을 포함한다: Face Swapping(10), Face Reenactment(13), Entire Face Synthesis(12), Face Editing(5).

영상 기반 딥페이크 탐지를 목표로 초기 실험은 **Face Swapping (FS)** 유형으로 범위를 제한한다.

| 용도 | 후보 기법 |
|---|---|
| 학습 | SimSwap / FaceSwap / FSGAN |
| 공통 Seen Test | SimSwap (학습에 사용하지 않은 별도 원본 영상) |
| 공통 Unseen Test | BlendFace, InSwapper (학습 및 모델 선택에 미사용) |

**유의사항:** 기법과 표본 수는 데이터 접근성을 확인한 뒤 확정한다. DF40에서 배포하는 전처리 이미지 중 일부는 Fake만 포함하므로 대응하는 Real 데이터와 원본 영상 ID를 별도로 확보·검증해야 한다.

초기에는 FF++ 출처를 통일해 **조작 기법 차이**를 우선 비교하고, 다른 출처(Celeb-DF)로의 일반화는 별도 확장 과제로 둔다.

## 3. Model & Pipeline

`Video → Frame Extraction / Face Cropping → Pretrained CNN → Frame-level Fake Score → (확장) Video-level Aggregation`

초기에는 DF40의 전처리된 얼굴 crop 이미지로 **Frame-level Binary Classification**을 수행한다. 사전학습된 이미지 분류기의 분류층을 Real/Fake 목적에 맞게 변경한 뒤 Fine-tuning한다.

| 모델 | 역할 | 진행 상태 |
|---|---|---|
| EfficientNet-B0 | 기본 CNN Baseline | 구현 예정 |
| ViT-B/16 | Transformer 모델 비교 | 선택적 확장 |
| Xception | DF40 및 기존 탐지 연구의 참고 Baseline | 선행연구 참고 |

**사용 AI 분야:** Computer Vision, Supervised Deep Learning, Transfer Learning / Fine-tuning, Domain Generalization.

## 4. Experimental Design

### Exp 1. CNN Baseline

- DF40 Face Swapping subset에서 EfficientNet-B0로 Real/Fake Fine-tuning
- 학습·검증 Loss와 프레임 단위 평가 지표 확인
- 동일한 모델에 대해 Seen과 Unseen 기법별 성능 기록

### Exp 2. Training Diversity — Core Experiment

**Fake 이미지 수 및 Real 이미지 수를 각각 고정**하고, 학습 Fake 이미지의 조작 기법 구성을 바꾼다.

| 조건 | Fake 학습 데이터 예시 | 총 Fake 수(예시) |
|---|---|---:|
| Case A — Single | SimSwap 3,000장 | 3,000 |
| Case B — Multiple | SimSwap 1,000장 + FaceSwap 1,000장 + FSGAN 1,000장 | 3,000 |

*수치는 설명을 위한 예시이며 확정된 데이터 규모가 아니다. 각 조건에 동일한 수의 Real 이미지도 사용한다.*

두 조건에서 **동일한 Seen Test(SimSwap)** 와 **동일한 Unseen Test(BlendFace, InSwapper)** 를 사용한다.

주요 결과는 기법별 ROC-AUC, F1, Accuracy와 Case A/B 간 성능 차이이다. Seen/Unseen 성능 차이도 참고하되, 테스트 기법의 난도 자체가 다르므로 차이를 단독으로 인과 해석하지 않는다.

### Exp 3. CNN vs ViT — Optional

동일한 데이터 분할과 평가 프로토콜을 유지하며 모델 구조에 따른 일반화 성능을 비교한다.

### Exp 4. Video-level Detection — Optional

동일 영상에서 추출한 프레임의 Fake 확률을 평균해 영상 단위 점수를 산출하고, 프레임별 점수 변화를 시각화한다.

프레임 점수는 조작 위치의 정답 라벨이 아니라 **모델의 예측 결과**임을 명시한다.

### Evaluation Protocol

1. 원본 영상 및 파생된 Fake 영상의 식별자를 기준으로 Train / Validation / Test 간 누수를 방지한다.
2. Case A/B의 **학습 Fake 수, Real 수, Real 원본 출처, 전처리 방식, 모델과 학습 설정**을 가능한 한 동일하게 유지한다. 원본 영상 수와 프레임 추출 간격도 기록한다.
3. 모델 선택·임계값 결정은 Validation에서만 하고, 공통 Unseen Test는 최종 평가에만 사용한다.
4. 공통 Seen Test는 두 Case가 모두 학습한 SimSwap의 **별도 영상**으로 구성한다.
5. ROC-AUC를 우선 보고하며 F1·Accuracy의 임계값과 클래스 비율을 함께 기록한다.
6. 가능하다면 여러 조작 기법 조합 및 3개 이상의 Random Seed로 반복하여 평균과 표준편차를 제시한다.
7. 결과는 '다양성의 순수한 인과효과'가 아니라 **고정 데이터 예산에서 선택한 학습 구성 전략의 비교 결과**로 해석한다.

## 5. Current Progress

| 항목 | 상태 |
|---|---|
| 프로젝트 주제 및 연구 질문 선정 | 완료 |
| DF40 데이터셋 및 모델 후보 선정 | 완료 |
| 실험 통제 조건과 평가 프로토콜 | 설계 초안 |
| GitHub README 및 개발 계획 | 문서화 |
| 실제 DF40 샘플 확보·로딩 | 미진행 |
| CNN Fine-tuning 및 성능 평가 | 미진행 |
| Training Diversity 실험 | 미진행 |
| ViT / Video-level 시각화 | 선택적 확장 |

## 6. Roadmap

| 단계 | 목표 | 확인할 산출물 |
|---|---|---|
| Step 1 | DF40 subset 접근 및 Real/Fake 매칭 검증 | 샘플 목록, 원본 영상 ID, DataLoader 출력 |
| Step 2 | CNN Baseline Fine-tuning | 실행 코드, Checkpoint, Loss 로그 |
| Step 3 | Seen / Unseen 평가 | 기법별 ROC-AUC·F1 결과표 |
| Step 4 | Single vs Multiple 학습 비교 | 성능 비교 그래프, Seen/Unseen Trade-off 분석 |
| Step 5 (선택) | ViT 비교 | 동일 조건 모델 비교표 |
| Step 6 (선택) | Video-level 시각화 | 영상 점수 및 프레임별 확률 그래프 |

**필수 목표: Step 1–4.** 데이터 및 GPU 확보 상황에 따라 Step 5–6을 확장한다.

## References

1. [DF40: Toward Next-Generation Deepfake Detection](https://github.com/YZY-stack/DF40) — NeurIPS 2024.
2. [Community Forensics: Using Thousands of Generators to Train Fake Image Detectors](https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html) — CVPR 2025.
3. [D³: Scaling Up Deepfake Detection by Learning from Discrepancy](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_D3_Scaling_Up_Deepfake_Detection_by_Learning_from_Discrepancy_CVPR_2025_paper.html) — CVPR 2025.
4. [Deepfake Detection that Generalizes Across Benchmarks (GenD)](https://cispa.de/en/research/publications/104500-deepfake-detection-that-generalizes-across-benchmarks) — WACV 2026.
5. [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) — 구현 및 평가 참고 프레임워크.

---

*2026-10-29 중간발표용 설계 초안. 실제 데이터 검증 및 구현 결과에 따라 갱신한다.*