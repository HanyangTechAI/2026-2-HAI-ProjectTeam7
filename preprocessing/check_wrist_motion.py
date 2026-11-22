import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from preprocess import load_video_keypoints, normalize_keypoints

video_ids = [
    "NIA_SL_SEN0001_REAL17_F",
    "NIA_SL_SEN0002_REAL17_F"
]

for video_id in video_ids:
    # 원본 키포인트 불러오기
    bodies, faces, left_hands, right_hands = load_video_keypoints(
        Path("data/keypoint/17"), video_id
    )

    # 정규화
    norm_bodies, _, _, _ = normalize_keypoints(
        bodies, faces, left_hands, right_hands
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, wrist_idx, title in [
        (axes[0], 4, "Right Wrist"),
        (axes[1], 7, "Left Wrist")
    ]:
        # 원본과 정규화된 손목 좌표
        original = bodies[:, wrist_idx, :2]
        normalized = norm_bodies[:, wrist_idx, :2]

        # 연속 프레임 사이 이동 거리
        original_motion = np.linalg.norm(
            np.diff(original, axis=0), axis=1
        )

        normalized_motion = np.linalg.norm(
            np.diff(normalized, axis=0), axis=1
        )

        print(f"\n{video_id} - {title}")
        print("원본 평균 이동 거리 (px):", original_motion.mean())
        print("정규화 후 평균 이동 거리:", normalized_motion.mean())
        print("원본 최대 이동 거리 (px):", original_motion.max())
        print("정규화 후 최대 이동 거리:", normalized_motion.max())

        ax.plot(
            np.arange(1, len(bodies)),
            normalized_motion
        )
        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Normalized Movement")
        ax.grid(True)

    fig.suptitle(video_id)
    plt.tight_layout()

plt.show()
