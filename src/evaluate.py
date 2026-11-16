"""한국수어 Manual-only/Fusion 평가 템플릿."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml", help="설정 YAML 경로 (로딩은 TODO)")
    parser.parse_args()
    # TODO: 설정·체크포인트 로딩, 추론, Accuracy·Macro-F1·클래스별 Precision/Recall·Confusion Matrix 저장
    parser.exit(status=1, message="미구현: 설정·체크포인트 로딩, 추론, Accuracy·Macro-F1·클래스별 Precision/Recall·Confusion Matrix 저장 기능을 구현해야 합니다.\n")


if __name__ == "__main__":
    main()
