"""데이터 manifest 생성 실행 템플릿."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml", help="설정 YAML 경로")
    parser.parse_args()
    # TODO: 데이터 탐색, 원본 영상 ID 기반 분할, 누수 검사 및 CSV 저장
    parser.exit(status=1, message="미구현: 데이터 탐색, 원본 영상 ID 기반 분할, 누수 검사 및 CSV 저장 기능을 구현해야 합니다.\n")


if __name__ == "__main__":
    main()
