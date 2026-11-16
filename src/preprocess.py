"""키포인트 파싱·정규화·시계열 변환 인터페이스 (미구현)."""


def preprocess(raw_dir, output_dir, sequence_length=32, keypoint_source="aihub"):
    """실제 JSON 스키마 확인 후 manual/fusion 시계열을 저장한다."""
    # TODO: 제공 JSON 파서 또는 추출 어댑터, 결측 처리, 정규화, 시간 리샘플링
    raise NotImplementedError("키포인트 전처리를 구현해야 합니다.")
