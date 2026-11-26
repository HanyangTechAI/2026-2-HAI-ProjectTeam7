import numpy as np
from missing import handle_missing_keypoints


#missing.py 검사용 파일


# 20프레임, 1개 키포인트, (x, y, confidence)
points = np.zeros((20, 1, 3))

# 정상 좌표 생성
for t in range(20):
    points[t, 0] = [100 + t * 10, 200 + t * 5, 1]

# 실제 정상 좌표 보존
original = points.copy()

# 인위적인 결측치 생성
points[0:2, 0] = 0      # 시작 부분 2프레임
points[4:6, 0] = 0      # 중간 2프레임
points[8:11, 0] = 0     # 중간 3프레임
points[12:16, 0] = 0    # 중간 4프레임
points[18:20, 0] = 0    # 끝부분 2프레임

# 결측치 처리
result, original_mask, valid_mask = handle_missing_keypoints(
    points,
    max_gap=3
)

# 결과 출력
for t in range(20):
    print(
        f"Frame {t:2d} | "
        f"Original: {original[t, 0]} | "
        f"Before: {points[t, 0]} | "
        f"After: {result[t, 0]} | "
        f"Original Mask: {original_mask[t, 0]} | "
        f"Valid Mask: {valid_mask[t, 0]}"
    )

# 예상 결과 자동 검증
expected_mask = np.ones((20, 1), dtype=bool)
expected_mask[0:2] = False
expected_mask[12:16] = False
expected_mask[18:20] = False

assert np.array_equal(valid_mask, expected_mask)
assert np.allclose(result[4:6, :, :2], original[4:6, :, :2])
assert np.allclose(result[8:11, :, :2], original[8:11, :, :2])
assert np.array_equal(result[original_mask], original[original_mask])
assert np.all(result[~valid_mask] == 0)
assert np.all(result[valid_mask & ~original_mask, 2] == 0)

print("\nAll tests passed!")
