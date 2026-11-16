"""Baseline 학습 실행 템플릿."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml", help="설정 YAML 경로")
    parser.parse_args()
    # TODO: 설정 로딩, seed 고정, DataLoader, 학습·검증 루프 및 체크포인트 저장
    parser.exit(status=1, message="미구현: 설정 로딩, seed 고정, DataLoader, 학습·검증 루프 및 체크포인트 저장 기능을 구현해야 합니다.\n")


if __name__ == "__main__":
    main()
