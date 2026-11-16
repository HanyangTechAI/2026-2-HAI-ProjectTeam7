# Manifest 계약 초안

실제 CSV는 아직 없습니다. 아래 필드는 팀 내부 목표 스키마이며 AI Hub JSON의 실제 필드명이 아닙니다. 데이터 확보 후 매핑을 문서화하고 생성합니다.

| 생성 예정 파일 | 컬럼 제안 | 역할 |
|---|---|---|
| class_map.csv | class_id, class_name | 0부터 연속된 정수 ID와 어휘 매핑 |
| manifest.csv | sample_id, video_path, keypoint_path, class_id, signer_id, recording_group_id, view_id | 원본과 라벨·수연자·촬영 그룹 연결 |
| split_manifest.csv | sample_id, split, processed_path | train/val/test와 전처리 결과 연결 |

모든 경로는 저장소 루트 기준 상대 경로이며 `sample_id`는 고유해야 합니다. 각 split 행은 manifest의 한 샘플과 일대일로 연결합니다. 같은 `signer_id`와 같은 `recording_group_id`는 split을 넘지 않아야 합니다. 수연자 또는 촬영 그룹을 확인할 수 없는 샘플은 임의로 추정해 분할하지 말고 별도로 보류합니다. 다각도 연결과 반복 촬영 식별 근거를 데이터 보고서에 기록합니다.

자동 검증 TODO: 필수값·중복·파일 존재·class_map 참조·분할 교집합·각 split 클래스 지원 여부 검사. CSV에는 개인 이름, 개인 PC 경로 및 민감한 원본 JSON 내용을 넣지 않습니다. 공개 가능성이 확인된 소규모 메타데이터만 Git에 추가합니다.
