# 한국수어 데이터 조사 및 실험 기록

현재 상태: 데이터 미확보·미분석, 학습·평가 미실행. 아래는 조사 템플릿이며 결과가 아닙니다.

## 데이터 출처와 이용 조건

- 대상: [AI Hub 수어영상 (103)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103)
- 확보 버전·일자·신청 및 이용 조건·재배포 범위: TODO
- 실제 사용 어휘 목록과 선정 근거 (10~30개 계획): TODO

## 실제 스키마 조사

- 영상·키포인트·라벨 연결 및 JSON 필드 매핑: TODO
- 수연자·동일 수행·촬영 각도 식별법: TODO
- 손·상체·얼굴 키포인트 수, 순서, 좌표계, 채널: TODO
- FPS, 결측·중복·손상·검출 실패 비율: TODO
- AI Hub 제공 키포인트와 재추출 필요성: TODO

## 데이터 분할

| 분할 | 클래스 수 | 샘플 수 | 수연자 수 | 수행 그룹 수 |
|---|---|---|---|---|
| Train | TBD | TBD | TBD | TBD |
| Validation | TBD | TBD | TBD | TBD |
| Test | TBD | TBD | TBD | TBD |

수연자·다각도·파생 샘플 교집합 검사, 클래스 지원 여부 및 클래스 불균형: TODO.

## 전처리 및 실험

T=32는 제안값입니다. 정규화·리샘플링·결측 처리·스키마 버전·입력 차원·모델 파라미터 수·seed·실행 환경을 기록합니다.

| 입력 | Accuracy | Macro-F1 | 클래스별 Precision/Recall | Confusion Matrix |
|---|---|---|---|---|
| Manual-only | TBD | TBD | TBD | TBD |
| Manual+Face | TBD | TBD | TBD | TBD |

오류 분석, 얼굴 특징 추가 효과와 한계: TODO. 실제 측정 전에는 수치를 채우지 않습니다.

## 저장소 재구성 확인 (2026-11-16)

- 변경 전: 딥페이크 미구현 템플릿 12개, src 패키지 초기화 파일·manifests·docs 없음.
- 변경 후: 기존 경로를 수어 템플릿으로 수정하고 src/__init__.py, manifests/README.md, docs/pipeline.md 추가. data/raw와 data/processed 로컬 빈 디렉터리 생성.
- 기존 12개 파일은 legacy/deepfake에 상대 구조 그대로 복사 보존. requirements.txt와 .gitignore는 변경하지 않음.
- 재사용: Dataset의 기본 메서드 구조, build_model/preprocess 진입점, --config CLI 구조. 영상 ID 이진 분류 규격은 수연자·수어 어휘 계약으로 변경.
- 검증 통과: 활성 Python 7개 문법, 의존성 없는 모듈 6개 import, CLI 3개 도움말 및 미구현 종료 코드 1, Git 제외 규칙.
- src/dataset.py는 torch 미설치로 import 실행 검증 불가. 외부 import는 torch.utils.data.Dataset이며 내부 파일 이동에 따른 import 참조는 없음.
- 실제 데이터 및 PyTorch 의존성 부재로 전처리·Tensor 배치·모델·학습·평가 테스트는 미실행. 구현 전 코드이므로 실행 가능하다고 주장하지 않음.
- Git 추적 파일은 README.md만 확인됨. 추적 중 데이터 삭제·추적 해제 없음. commit/push 미실행.
