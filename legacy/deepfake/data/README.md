# 데이터 안내

원본 데이터는 `data/raw/`, 전처리 이미지와 manifest는 `data/processed/`에 저장합니다.
두 디렉터리는 데이터 확보 및 전처리 단계에서 생성하며 Git에서 제외합니다.

## Manifest 형식 (예정)

CSV 컬럼: `path,label,video_id,method,split`

- `path`: 저장소 루트 기준 이미지 경로
- `label`: Real은 0, Fake는 1
- `video_id`: 분할 누수 검증을 위한 원본 영상 식별자
- `method`: real 또는 조작 기법 이름
- `split`: train, val, test

동일 원본 영상에서 파생된 모든 데이터는 같은 split에 배정합니다.
여러 원본 영상이 연결된 조작 데이터는 관련 원본 ID도 함께 검증해야 합니다.
데이터 출처, 규모, 분할 기준과 검증 결과는 `reports/data_report.md`에 기록합니다.
