# HAI-Team7

DF40 데이터를 활용한 딥페이크 탐지 프로젝트의 초기 템플릿입니다.
학습 데이터의 조작 기법 다양성이 미학습 기법 탐지 성능에 미치는 영향을 비교합니다.

## 파일 구조

```text
HAI-Team7/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── baseline.yaml
├── data/
│   └── README.md
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── scripts/
│   └── prepare_manifest.py
└── reports/
    └── data_report.md
```

## 환경 준비

Python 3.10 이상을 사용합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch의 GPU 환경은 사용하는 서버의 CUDA 환경에 맞게 구성합니다.

## 구현 순서

1. `scripts/prepare_manifest.py`: 원본 영상 ID 기반 데이터 분할 및 manifest 생성
2. `src/preprocess.py`: 프레임 추출과 얼굴 이미지 전처리
3. `src/dataset.py`: manifest 기반 Dataset 구현
4. `src/model.py`: Real/Fake 분류 모델 구성
5. `src/train.py`: baseline 학습과 체크포인트 저장
6. `src/evaluate.py`: Seen/Unseen 평가

설정은 `configs/baseline.yaml`, 데이터 조사 기록은 `reports/data_report.md`에 작성합니다.
현재 Python 파일은 인터페이스와 TODO만 포함하며, 실제 전처리·학습·평가는 아직 구현하지 않았습니다.
