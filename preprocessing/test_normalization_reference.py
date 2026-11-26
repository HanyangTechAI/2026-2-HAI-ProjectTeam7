import numpy as np
from preprocess import repair_normalization_reference


def test_reference_repair():
    T = 30

    # 정상적인 정규화 기준값 생성
    center = np.array([
        [900 + 2 * t, 400 + t]
        for t in range(T)
    ], dtype=float)

    scale = np.array([
        280 + t for t in range(T)
    ], dtype=float)

    # 테스트할 결측 구간
    test_cases = [
        ("Normal", 0, 0),
        ("Middle 4 frames", 5, 9),
        ("Middle 5 frames", 5, 10),
        ("Middle 6 frames", 5, 11),
        ("Start 3 frames", 0, 3),
        ("End 3 frames", 27, 30),
        ("All frames", 0, 30)
    ]

    for name, start, end in test_cases:
        damaged_center = center.copy()
        damaged_scale = scale.copy()

        shoulder_valid = np.ones(T, dtype=bool)

        # 인위적인 결측치 생성
        damaged_center[start:end] = 0
        damaged_scale[start:end] = 0
        shoulder_valid[start:end] = False

        # 정규화 기준값 복원
        result_center, result_scale, reference_valid = (
            repair_normalization_reference(
                damaged_center,
                damaged_scale,
                shoulder_valid,
                max_gap=5
            )
        )

        # 예상 Mask 생성
        expected_mask = shoulder_valid.copy()

        if end - start <= 5:
            expected_mask[start:end] = True

        # Mask 검증
        assert np.array_equal(
            reference_valid, expected_mask
        )

        # 기존 정상 기준값 유지 검증
        assert np.allclose(
            result_center[shoulder_valid],
            center[shoulder_valid]
        )

        assert np.allclose(
            result_scale[shoulder_valid],
            scale[shoulder_valid]
        )

        # 중간 결측 선형 보간 검증
        if 0 < start < end < T and end - start <= 5:
            assert np.allclose(
                result_center[start:end],
                center[start:end]
            )

            assert np.allclose(
                result_scale[start:end],
                scale[start:end]
            )

        # 시작 부분 복원 검증
        elif start == 0 and 0 < end <= 5:
            assert np.allclose(
                result_center[start:end], center[end]
            )

            assert np.allclose(
                result_scale[start:end], scale[end]
            )

        # 끝부분 복원 검증
        elif end == T and start > 0 and end - start <= 5:
            assert np.allclose(
                result_center[start:end], center[start - 1]
            )

            assert np.allclose(
                result_scale[start:end], scale[start - 1]
            )

        print(f"\n=== {name} ===")
        print("Missing Frames:", end - start)
        print("Valid Before:", np.sum(shoulder_valid))
        print("Valid After:", np.sum(reference_valid))
        print("Test Passed: True")

    print("\nAll reference repair tests passed!")


if __name__ == "__main__":
    test_reference_repair()