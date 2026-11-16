# 팀 간 데이터 인터페이스

이 문서는 구현 전 계약 초안입니다. 실제 AI Hub JSON 필드와 키포인트 규격은 아직 분석하지 않았습니다. 변경 시 데이터·전처리·모델 담당자가 함께 설정과 문서를 갱신합니다.

## 1. 데이터 담당 → 전처리 담당

`manifests/class_map.csv`, `manifest.csv`, `split_manifest.csv`를 전달합니다. 컬럼은 [Manifest 문서](../manifests/README.md)를 따릅니다. 데이터 담당자는 원본 필드에서 내부 컬럼으로의 매핑, 클래스 선정 기준, 수연자와 다각도 그룹 식별 근거를 보고서에 작성합니다. 모든 경로는 저장소 루트 기준 상대 경로입니다. 실행은 저장소 루트에서 수행하며 다른 작업 경로 지원은 추후 구현합니다.

수연자별 Train/Validation/Test를 먼저 확정합니다. 동일 수행의 모든 촬영 각도·중복·파생 클립은 같은 그룹에 둡니다. 수연자 ID가 데이터 일부에서 재사용되는지 확인하고 신원을 추정하지 않습니다. 모든 split에서 클래스가 평가 가능한지 확인하되 클래스 수를 맞추려고 동일 수연자를 다른 split에 넣지 않습니다. 데이터 부족 시 클래스를 재선정하고 근거를 남깁니다.

## 2. 전처리 담당 → 모델 담당

Dataset의 `__getitem__` 목표 반환값:

```python
{
    "manual": manual_tensor,  # float32 (T, J_manual, C)
    "fusion": fusion_tensor,  # float32 (T, J_fusion, C)
    "label": class_id,        # 정수, 0 <= class_id < num_classes
}
```

`T=32`는 초기 제안이며 `data.sequence_length`에서 변경합니다. DataLoader의 목표 배치는 `manual: (B,T,J_manual,C)`, `fusion: (B,T,J_fusion,C)`, `label: (B,)`이고 라벨은 `torch.long`입니다. **J와 C는 실제 JSON 분석 후 확정**합니다. confidence/visibility는 좌표와 같은 값으로 정규화하지 않고, 별도 마스크 또는 메타데이터 처리 방식을 합의합니다.

Fusion은 Manual의 관절 순서와 값을 그대로 앞부분에 두고 얼굴·머리 특징을 추가하는 규격을 제안합니다. 관절 목록·좌표계·단위·채널 순서를 schema_version별로 고정해야 합니다. 두 입력은 같은 시점과 같은 리샘플링 인덱스를 사용합니다. 시계열 길이 변경, 결측 보간·마스킹, 패딩 정책은 TODO이며 전체 결측·0길이·잘못된 라벨은 명시적으로 거부해야 합니다.

학습 전에 원본 좌표를 Tensor로 변환해 저장합니다. 샘플별 `.npz`에 `manual`, `fusion`, `label`을 저장하는 방식을 제안하며 `processed_path`로 연결합니다. 이 저장·로딩 기능은 아직 구현하지 않았습니다. 스키마 버전과 정규화 정책은 전처리 산출물 메타데이터로 함께 관리합니다. 모델 학습은 전처리된 Tensor, class_map, split_manifest만 읽도록 하여 원본 영상 및 추출 도구가 필요하지 않게 합니다.

## 3. 정규화와 추출 도구

AI Hub 제공 JSON과 MediaPipe 재추출 데이터는 관절 번호·얼굴 구성·좌표계·confidence·프레임 시각이 다를 수 있습니다. 제공 키포인트 파서와 영상 추출 어댑터를 분리하고 공통 규격으로 명시적으로 매핑합니다. MediaPipe의 관절 수를 AI Hub 규격으로 가정하거나 서로 다른 추출 결과를 무표시 혼합하지 않습니다. 추출 도구·버전·매핑을 기록합니다.

몸 기준 위치·크기 정규화를 검토하되 얼굴을 프레임마다 독립적으로 회전 정렬해 머리 움직임을 지워버리지 않도록 합니다. 머리 방향이나 움직임을 별도 특징으로 계산할 경우 산출법과 차원을 문서화합니다. 데이터 기반 정규화 통계는 Train에서만 계산하고 Validation/Test에 고정 적용합니다.

## 4. 모델·학습·평가 계약

`build_model`은 설정의 `gru` 또는 `cnn1d`에 대응할 예정이며 입력 `(B,T,J,C)`를 `(B,T,J*C)`로 변환하는 위치를 모델 내부로 통일합니다. 출력은 `(B,K)`의 정규화 전 logits입니다. `CrossEntropyLoss`에 정수 라벨을 전달하며 softmax는 추론 확률 계산에만 사용합니다. 현재 모델은 미구현입니다.

`experiment: manual`은 배치의 manual만, `fusion`은 fusion만 사용합니다. 같은 split·seed·훈련 예산을 유지하고 각 실행의 설정·전처리 버전·class_map·체크포인트를 함께 기록합니다. 검증 Macro-F1 등 모델 선택 기준은 구현 시 확정합니다. 테스트에서는 Accuracy, Macro-F1, 고정된 전체 클래스 목록 기준 클래스별 Precision/Recall과 Confusion Matrix를 기록합니다. 검증·테스트에 없는 클래스와 zero-division 처리 정책도 명시합니다.

## 5. 서비스 및 확장

서비스 담당은 영상 입력을 동일 전처리 규격으로 변환하고 모델 logits에서 class_map 기반 예측 어휘와 확률을 표시하는 인터페이스를 협의합니다. 실시간 영상 추출·서비스는 미구현입니다. 보조 비수지 라벨과 Multi-task head는 향후 확장이며 현재 필수 반환값을 임의로 변경하지 않습니다.
