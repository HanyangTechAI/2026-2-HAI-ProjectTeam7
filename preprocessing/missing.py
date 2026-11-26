import numpy as np

def get_valid_mask(
    points,
    width=1920,
    height=1080,
    confidence_threshold=0.5
):
    x = points[:, :, 0]
    y = points[:, :, 1]
    confidence = points[:, :, 2]

    # NaN, Inf 검사
    finite = np.isfinite(points).all(axis=2)

    # (0, 0) 좌표 검사
    nonzero = ~((x == 0) & (y == 0))

    # 이미지 범위 검사
    in_bounds = (
        (x >= 0) & (x < width) &
        (y >= 0) & (y < height)
    )

    # Confidence 검사
    confident = (
        (confidence > 0) &
        (confidence >= confidence_threshold) &
        (confidence <= 1)
    )

    # 모든 조건을 만족해야 정상 좌표
    valid_mask = (
        finite &
        nonzero &
        in_bounds &
        confident
    )

    return valid_mask

def handle_missing_keypoints(
    points,
    max_gap=3,
    width=1920,
    height=1080,
    confidence_threshold=0.5
):
    # 원본 데이터 보존
    result = points.astype(float).copy()

    original_mask = get_valid_mask(
        points,
        width,
        height,
        confidence_threshold
    )

    valid_mask = original_mask.copy()

    T, K, C = points.shape

    # 각 키포인트별로 검사
    for k in range(K):
        t = 0

        while t < T:

            # 정상 좌표는 건너뛰기
            if original_mask[t, k]:
                t += 1
                continue

            # 결측 구간 시작
            start = t

            # 연속된 결측 구간 탐색
            while t < T and not original_mask[t, k]:
                t += 1

            end = t
            gap = end - start

            # 선형 보간 가능 여부 확인
            if start > 0 and end < T and gap <= max_gap:

                left = result[start - 1, k, :2]
                right = result[end, k, :2]

                # Linear Interpolation
                for i in range(gap):
                    ratio = (i + 1) / (gap + 1)

                    result[start + i, k, :2] = (
                        left + ratio * (right - left)
                    )

                valid_mask[start:end, k] = True

                # 실제 검출된 좌표가 아님을 표시
                result[start:end, k, 2] = 0

            else:
                # 복원 불가능한 좌표는 0으로 채움
                result[start:end, k, :] = 0

                valid_mask[start:end, k] = False

    return result, original_mask, valid_mask
