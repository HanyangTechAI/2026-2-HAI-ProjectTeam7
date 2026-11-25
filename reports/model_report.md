# 2026-11-19 작업 보고: GRU 수어 어휘 분류 모델

## 작업 목표

32프레임 키포인트 시계열을 입력받아 수어 어휘 클래스를 분류하는 GRU baseline 구현

## 진행 내용

- PyTorch `nn.GRU`를 사용해 양방향 GRU 기반 `GRUClassifier` 구현
- 입력 `(B, T, F)`의 차원 및 특징 수 `F`가 `input_size`와 일치하는지 검사
- 마지막 GRU 층의 정방향·역방향 hidden state를 연결해 영상 단위 특징 생성
- Dropout과 Linear 분류층을 통해 `(B, num_classes)` 크기의 클래스별 logits 출력
- `build_model()`에서 입력 특징 수, hidden size, GRU 층 수, 클래스 수, dropout 설정 가능
- 더미 데이터로 Forward 및 Backward 실행 확인

## 테스트 결과

설정: 배치 크기 4, 프레임 수 32, 프레임당 특징 수 126, hidden size 128, 클래스 수 10

```text
출력 크기: torch.Size([4, 10])
손실: 2.376555919647217
GRU 가중치 기울기: torch.Size([384, 126])
Forward / Backward 테스트 완료!
```

- 영상 4개에 대해 각각 10개 클래스 점수가 출력되고, GRU 가중치의 기울기가 계산됨을 확인함.
- 무작위 입력과 라벨을 사용한 동작 확인이며, 실제 수어 인식 성능을 평가한 결과는 아님.

## 현재 상태 및 다음 작업

- 현재는 양방향 GRU를 사용하며 `build_model()`은 GRU만 지원함. 1D CNN은 미구현.
- 실제 데이터의 키포인트 구성과 클래스 수를 확인하고 입력 특징 수를 결정할 예정.
- 실제 데이터 전처리, 학습·검증 및 성능 평가가 필요함.
- 실시간 추론 방식 검토 필요: 최근 32프레임을 모아 예측할 경우 현재 양방향 모델을 사용할 수 있음. 프레임마다 hidden state를 이어받는 방식은 단방향 모델과 별도 추론 구조가 필요함.
- 웹캠 입력, 키포인트 추출 및 실시간 예측 연결은 현재 모델 코드에 포함되지 않음.


## 2026-11-25 작업 내역

### 완료한 작업

- GRU 출력으로 Gloss 클래스를 예측하는 코드 확인
- CrossEntropyLoss 및 AdamW Optimizer 적용 확인
- Training/Validation Loop 구현 확인
- 더미 데이터로 2 epoch 학습·검증 실행
- Forward/Backward 및 파라미터 갱신 과정이 오류 없이 실행되는 것 확인

### 테스트 환경 및 결과

- 실행 장치: CPU
- 입력 크기: `(T=32, F=126)`
- 클래스 수: 10개
- 실행 명령어: `python -m src.train --dummy --epochs 2`

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
|---|---:|---:|---:|---:|
| 1 | 2.2860 | 0.125 | 2.3262 | 0.125 |
| 2 | 2.1185 | 0.547 | 2.3498 | 0.000 |

> 위 결과는 무작위 더미 데이터로 학습·검증한 결과이며, 실제 수어 인식 성능을 의미하지 않습니다.

### 남은 작업

- AI Hub 수어 데이터 이용 조건 및 파일 구조 확인
- 실제 데이터 확보 후 키포인트와 Gloss 라벨 연결
- 실제 데이터로 학습·검증 및 평가 진행