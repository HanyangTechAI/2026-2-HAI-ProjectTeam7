"""AI Hub 수어 데이터 Manifest 생성 템플릿."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml", help="설정 YAML 경로 (로딩은 TODO)")
    parser.parse_args()
    # TODO: 실제 JSON 분석, 클래스 매핑, 수연자·다각도 그룹 분할, 누수 검증 및 CSV 저장
    parser.exit(status=1, message="미구현: 실제 JSON 분석, 클래스 매핑, 수연자·다각도 그룹 분할, 누수 검증 및 CSV 저장 기능을 구현해야 합니다.\n")


if __name__ == "__main__":
    main()
