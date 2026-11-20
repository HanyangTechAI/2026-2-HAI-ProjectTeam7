# 한국 수어 데이터 전처리

## 1. 프로젝트 소개

AI Hub 103번 한국 수어 영상 데이터의 키포인트 JSON 파일을 AI 모델 학습에 사용할 수 있는 NumPy 배열로 변환하는 전처리 프로그램입니다.

영상의 각 프레임에서 몸, 양손, 얼굴의 키포인트를 추출하고, Manual-only와 Manual+Face 두 가지 입력 형태로 구분하여 NPZ 파일로 저장합니다.

또한 PyTorch Dataset을 이용하여 저장된 NPZ 파일을 Tensor 형태로 불러올 수 있습니다.

## 2. 주요 파일 설명

- `preprocess.py`: 키포인트 JSON을 읽고, 영상별 프레임을 결합하여 NPZ 파일로 저장합니다.
- `dataset.py`: NPZ 파일을 불러와 PyTorch Tensor로 변환합니다.
- `data/`: 테스트에 사용하는 원본 키포인트 JSON 파일이 저장되는 폴더입니다.
- `outputs/`: 전처리 결과인 NPZ 파일이 저장되는 폴더입니다.

## 3. 실행 환경 및 라이브러리

본 전처리 프로그램은 Python 환경에서 실행합니다.

사용하는 주요 라이브러리는 다음과 같습니다.

- `numpy`: 키포인트 좌표 배열 처리 및 NPZ 파일 저장
- `torch`: NPZ 데이터를 PyTorch Tensor로 변환

필요한 라이브러리는 터미널에서 다음 명령어로 설치할 수 있습니다.

```bash
python -m pip install numpy torch
```

`json`, `pathlib`는 Python 표준 라이브러리이므로 별도 설치가 필요하지 않습니다.

## 4. 입력 데이터 및 폴더 구조

본 프로그램은 AI Hub 103번 한국 수어 영상 데이터의 프레임별 키포인트 JSON 파일을 입력으로 사용합니다.

테스트에 사용한 영상은 다음과 같습니다.

- `NIA_SL_SEN0001_REAL17_F`: 143프레임
- `NIA_SL_SEN0002_REAL17_F`: 144프레임

각 영상의 JSON 파일은 `data/keypoint/17/` 아래에 영상 ID별 폴더로 저장합니다.

```text
preprocessing/
├── preprocess.py
├── dataset.py
├── README.md
├── data/
│   └── keypoint/
│       └── 17/
│           ├── NIA_SL_SEN0001_REAL17_F/
│           │   └── *_keypoints.json
│           └── NIA_SL_SEN0002_REAL17_F/
│               └── *_keypoints.json
└── outputs/
    ├── NIA_SL_SEN0001_REAL17_F.npz
    └── NIA_SL_SEN0002_REAL17_F.npz
```

원본 JSON 파일은 영상의 각 프레임에서 추출한 몸, 양손, 얼굴의 2차원 키포인트 좌표 `(x, y, confidence)`를 포함합니다.

`outputs/` 폴더에는 전처리 과정에서 생성된 NPZ 파일이 저장됩니다.

## 5. 실행 방법

모든 명령어는 `preprocessing/` 폴더에서 실행합니다.

### 5.1. JSON을 NPZ로 변환

`data/keypoint/17/` 폴더에 원본 키포인트 JSON 파일을 준비한 뒤 다음 명령어를 실행합니다.

```bash
python preprocess.py
```

실행하면 각 영상의 프레임별 키포인트를 읽어 `outputs/` 폴더에 NPZ 파일을 생성합니다.

테스트 결과 다음 두 파일이 생성되었습니다.

- `NIA_SL_SEN0001_REAL17_F.npz`
- `NIA_SL_SEN0002_REAL17_F.npz`

### 5.2. PyTorch Dataset 테스트

NPZ 파일 생성이 완료되면 다음 명령어를 실행합니다.

```bash
python dataset.py
```

`dataset.py`는 `outputs/` 폴더의 NPZ 파일을 읽어 PyTorch Tensor로 변환합니다.

Manual-only 모드로 실행한 테스트 결과는 다음과 같습니다.

```text
영상 개수: 2
데이터 타입: <class 'torch.Tensor'>
Tensor shape: torch.Size([143, 67, 3])
Tensor dtype: torch.float32
```

## 6. NPZ 출력 데이터 구조

각 NPZ 파일에는 두 가지 키포인트 배열이 저장됩니다.

| 배열 이름 | 구성 | Shape |
|---|---|---|
| `manual` | 몸 25개 + 왼손 21개 + 오른손 21개 | `(T, 67, 3)` |
| `manual_face` | 몸 25개 + 양손 42개 + 얼굴 70개 | `(T, 137, 3)` |

여기서 각 차원의 의미는 다음과 같습니다.

- `T`: 영상의 전체 프레임 수
- `67` 또는 `137`: 프레임당 키포인트 개수
- `3`: 각 키포인트의 `(x, y, confidence)` 값

현재 좌표는 원본 JSON의 2차원 픽셀 좌표이며, 별도의 좌표 정규화는 적용하지 않았습니다.

실제 테스트 영상의 출력 Shape는 다음과 같습니다.

| 영상 | `manual` | `manual_face` |
|---|---|---|
| `SEN0001_REAL17_F` | `(143, 67, 3)` | `(143, 137, 3)` |
| `SEN0002_REAL17_F` | `(144, 67, 3)` | `(144, 137, 3)` |

현재 NPZ에는 키포인트 입력 데이터만 저장되어 있으며, Gloss/NMS 정답 라벨은 아직 연결되지 않았습니다.
