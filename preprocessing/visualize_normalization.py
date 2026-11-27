import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from preprocess import load_video_keypoints, normalize_keypoints

video_id = "NIA_SL_SEN0001_REAL17_F"

# 1. 원본 키포인트 불러오기
bodies, faces, left_hands, right_hands = load_video_keypoints(
    Path("data/keypoint/17"), video_id
)

# 2. 정규화 적용
norm_body, norm_face, norm_left, norm_right = normalize_keypoints(
    bodies, faces, left_hands, right_hands
)

# 3. 비교할 프레임 선택
frame_indices = [0, 50, 100]

# 4. 정규화 전후 비교 그래프 생성
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for col, frame_idx in enumerate(frame_indices):

    original = np.concatenate(
        [
            bodies[frame_idx],
            left_hands[frame_idx],
            right_hands[frame_idx],
            faces[frame_idx]
        ],
        axis=0
    )

    normalized = np.concatenate(
        [
            norm_body[frame_idx],
            norm_left[frame_idx],
            norm_right[frame_idx],
            norm_face[frame_idx]
        ],
        axis=0
    )

    # 위쪽: 정규화 전
    axes[0, col].scatter(
        original[:, 0], original[:, 1], s=8
    )
    axes[0, col].set_title(f"Frame {frame_idx} - Before")

    # 아래쪽: 정규화 후
    axes[1, col].scatter(
        normalized[:, 0], normalized[:, 1], s=8
    )
    axes[1, col].set_title(f"Frame {frame_idx} - After")

# 5. 모든 그래프에 동일한 축 설정
for row in axes:
    for ax in row:
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.grid(True)

plt.tight_layout()
plt.show()
