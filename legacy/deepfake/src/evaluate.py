"""Seen/Unseen 평가 실행 템플릿."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml", help="설정 YAML 경로")
    parser.parse_args()
    # TODO: 체크포인트 로딩, 추론, ROC-AUC·F1·Accuracy 계산 및 결과 저장
    parser.exit(status=1, message="미구현: 체크포인트 로딩, 추론, ROC-AUC·F1·Accuracy 계산 및 결과 저장 기능을 구현해야 합니다.\n")


if __name__ == "__main__":
    main()
