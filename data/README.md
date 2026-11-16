# AI Hub 수어영상 데이터

대상: [AI Hub 수어영상 데이터셋 (103)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=103).
공식 페이지의 신청·다운로드 절차, 이용 조건, 개인정보 및 원본 재배포 제한을 확인한 뒤 확보합니다. 이 저장소에는 실제 데이터가 없습니다.

- `raw/`: 승인받아 확보한 영상, 키포인트 JSON, 어노테이션 원본. 제공된 디렉터리와 파일명은 보존합니다.
- `processed/`: 정규화·시간 길이 조정 후 키포인트 시계열. 저장 규격 제안은 `../docs/pipeline.md` 참조.
- `../manifests/`: 공개 가능한 소규모 메타데이터와 스키마. 수연자 정보는 비식별 ID만 사용하며 업로드 전 공개 가능 여부를 확인합니다.

`raw/`, `processed/`는 Git에서 제외됩니다. 새 clone에서 필요하면 다음과 같이 생성합니다.

```bash
mkdir -p data/raw data/processed
```

수어 어휘 10~30개 선정은 계획이며 실제 클래스·샘플 수는 미확인입니다. JSON 필드명, 관절 수, 채널, 단위, 누락값 및 수연자·촬영 각도 식별법을 조사해 `../reports/data_report.md`에 기록합니다. AI Hub 제공 키포인트 우선 사용 여부를 결정하고, MediaPipe 재추출 결과와 같은 규격이라고 가정하지 않습니다.
