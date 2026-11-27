import numpy as np
from pathlib import Path
from preprocess import load_video_keypoints
from missing import handle_missing_keypoints

#3프레임 단위로 쪼갠 것이 적합한지 보기위해 1,2,3,5 프레임에 대해서 기존의 점
#제거하고 오차를 분석하기
#선형보간을 얼마나 잘하는지 보기

video_ids = [
    "NIA_SL_SEN0001_REAL17_F",
    "NIA_SL_SEN0002_REAL17_F"
]

data_dir = Path("data/keypoint/17")

gap_sizes = [1, 2, 3, 5]

def test_interpolation_error():
    for video_id in video_ids:

        bodies, faces, left_hands, right_hands = \
            load_video_keypoints(data_dir, video_id)

        print(f"\n=== {video_id} ===")

        for name, points in [
            ("Body", bodies),
            ("Face", faces),
            ("Left Hand", left_hands),
            ("Right Hand", right_hands)
        ]:

            print(f"\n{name}")

            for gap in gap_sizes:

                errors = []

                # 여러 구간에서 보간 테스트
                for start in range(1, len(points) - gap, 10):

                    end = start + gap

                    # 원본 데이터 복사
                    damaged = points.copy()

                    # 인위적인 결측치 생성
                    damaged[start:end, :, :] = 0

                    # 선형 보간
                    result, original_mask, valid_mask = \
                        handle_missing_keypoints(
                            damaged,
                            max_gap=gap
                        )

                    # 원본 좌표와 복원 좌표 비교
                    diff = (
                        result[start:end, :, :2]
                        - points[start:end, :, :2]
                    )

                    # 유클리드 거리 계산
                    distance = np.linalg.norm(diff, axis=2)

                    errors.extend(distance.flatten())

                print(
                    f"Gap {gap} frames | "
                    f"Mean Error: {np.mean(errors):.3f} px | "
                    f"Max Error: {np.max(errors):.3f} px"
                )

def test_data_preservation():
    print("\n=== Normal Data Preservation Test ===")

    for video_id in video_ids:

        bodies, faces, left_hands, right_hands = \
            load_video_keypoints(data_dir, video_id)

        print(f"\n{video_id}")

        for name, points in [
            ("Body", bodies),
            ("Face", faces),
            ("Left Hand", left_hands),
            ("Right Hand", right_hands)
        ]:

            result, original_mask, valid_mask = \
                handle_missing_keypoints(
                    points,
                    max_gap=3
                )

            # 원본 좌표 유지 여부
            unchanged = np.array_equal(points, result)

            # 모든 좌표가 정상인지 확인
            original_valid = np.all(original_mask)
            processed_valid = np.all(valid_mask)

            print(f"\n{name}")
            print("Original Preserved:", unchanged)
            print("Original Mask All True:", original_valid)
            print("Valid Mask All True:", processed_valid)

            assert unchanged
            assert original_valid
            assert processed_valid

    print("\nAll preservation tests passed!")

if __name__ == "__main__":
    #test_interpolation_error()
    test_data_preservation()