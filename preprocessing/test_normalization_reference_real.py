import numpy as np
from pathlib import Path
from preprocess import (
    load_video_keypoints,
    repair_normalization_reference
)

video_ids = [
    "NIA_SL_SEN0001_REAL17_F",
    "NIA_SL_SEN0002_REAL17_F"
]

data_dir = Path("data/keypoint/17")
gap_sizes = [1, 2, 3, 4, 5, 6]


def test_reference_error():
    for video_id in video_ids:
        bodies, faces, left_hands, right_hands = (
            load_video_keypoints(data_dir, video_id)
        )

        # 원래 정규화 기준값 계산
        right_shoulder = bodies[:, 2, :2]
        left_shoulder = bodies[:, 5, :2]

        center = (right_shoulder + left_shoulder) / 2

        scale = np.linalg.norm(
            right_shoulder - left_shoulder,
            axis=1
        )

        T = len(scale)

        # 비교할 신체 부위
        groups = [
            ("Body", bodies),
            ("Face", faces),
            ("Left Hand", left_hands),
            ("Right Hand", right_hands)
        ]

        print(f"\n=== {video_id} ===")

        for gap in gap_sizes:
            center_errors = []
            scale_errors = []

            normalization_errors = {
                name: [] for name, points in groups
            }

            for start in range(1, T - gap, 10):
                end = start + gap

                # 인위적인 정규화 기준값 결측 생성
                damaged_center = center.copy()
                damaged_scale = scale.copy()
                shoulder_valid = np.ones(T, dtype=bool)

                damaged_center[start:end] = 0
                damaged_scale[start:end] = 0
                shoulder_valid[start:end] = False

                # 정규화 기준값 복원
                repaired_center, repaired_scale, reference_valid = (
                    repair_normalization_reference(
                        damaged_center,
                        damaged_scale,
                        shoulder_valid,
                        max_gap=gap
                    )
                )

                assert np.all(reference_valid[start:end])

                # 어깨 중심 복원 오차
                center_diff = (
                    repaired_center[start:end]
                    - center[start:end]
                )

                center_errors.extend(
                    np.linalg.norm(center_diff, axis=1)
                )

                # 어깨 너비 복원 오차
                scale_errors.extend(
                    np.abs(
                        repaired_scale[start:end]
                        - scale[start:end]
                    )
                )

                # 신체 부위별 정규화 오차
                for name, points in groups:
                    xy = points[start:end, :, :2]

                    original_normalized = (
                        xy - center[start:end, None, :]
                    ) / scale[start:end, None, None]

                    repaired_normalized = (
                        xy - repaired_center[start:end, None, :]
                    ) / repaired_scale[start:end, None, None]

                    diff = (
                        repaired_normalized
                        - original_normalized
                    )

                    distance = np.linalg.norm(diff, axis=2)

                    normalization_errors[name].extend(
                        distance.flatten()
                    )

            print(f"\n--- Gap {gap} frames ---")

            print(
                f"Center Error | "
                f"Mean: {np.mean(center_errors):.3f} px | "
                f"Max: {np.max(center_errors):.3f} px"
            )

            print(
                f"Scale Error  | "
                f"Mean: {np.mean(scale_errors):.3f} px | "
                f"Max: {np.max(scale_errors):.3f} px"
            )

            for name, errors in normalization_errors.items():
                print(
                    f"{name:10s} Normalization Error | "
                    f"Mean: {np.mean(errors):.6f} | "
                    f"Max: {np.max(errors):.6f}"
                )

def test_boundary_reference_error():
    for video_id in video_ids:
        bodies, faces, left_hands, right_hands = (
            load_video_keypoints(data_dir, video_id)
        )

        right_shoulder = bodies[:, 2, :2]
        left_shoulder = bodies[:, 5, :2]

        center = (right_shoulder + left_shoulder) / 2

        scale = np.linalg.norm(
            right_shoulder - left_shoulder,
            axis=1
        )

        T = len(scale)

        groups = [
            ("Body", bodies),
            ("Face", faces),
            ("Left Hand", left_hands),
            ("Right Hand", right_hands)
        ]

        print(f"\n=== {video_id} ===")

        for gap in range(1, 5):
            cases = [
                ("Start", 0, gap),
                ("End", T - gap, T)
            ]

            for name, start, end in cases:
                damaged_center = center.copy()
                damaged_scale = scale.copy()
                shoulder_valid = np.ones(T, dtype=bool)

                damaged_center[start:end] = 0
                damaged_scale[start:end] = 0
                shoulder_valid[start:end] = False

                repaired_center, repaired_scale, reference_valid = (
                    repair_normalization_reference(
                        damaged_center,
                        damaged_scale,
                        shoulder_valid,
                        max_gap=4
                    )
                )

                assert np.all(reference_valid[start:end])

                center_error = np.linalg.norm(
                    repaired_center[start:end]
                    - center[start:end],
                    axis=1
                )

                scale_error = np.abs(
                    repaired_scale[start:end]
                    - scale[start:end]
                )

                print(f"\n--- {name}: Gap {gap} frames ---")

                print(
                    f"Center Error | "
                    f"Mean: {np.mean(center_error):.3f} px | "
                    f"Max: {np.max(center_error):.3f} px"
                )

                print(
                    f"Scale Error | "
                    f"Mean: {np.mean(scale_error):.3f} px | "
                    f"Max: {np.max(scale_error):.3f} px"
                )

                for group_name, points in groups:
                    xy = points[start:end, :, :2]

                    original_normalized = (
                        xy - center[start:end, None, :]
                    ) / scale[start:end, None, None]

                    repaired_normalized = (
                        xy - repaired_center[start:end, None, :]
                    ) / repaired_scale[start:end, None, None]

                    distance = np.linalg.norm(
                        repaired_normalized - original_normalized,
                        axis=2
                    )

                    print(
                        f"{group_name:10s} Normalization Error | "
                        f"Mean: {np.mean(distance):.6f} | "
                        f"Max: {np.max(distance):.6f}"
                    )

if __name__ == "__main__":
    #test_reference_error()
    test_boundary_reference_error()