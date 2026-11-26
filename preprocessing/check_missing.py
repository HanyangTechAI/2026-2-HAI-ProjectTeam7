import numpy as np
from pathlib import Path
from preprocess import load_video_keypoints

video_ids = [
    "NIA_SL_SEN0001_REAL17_F",
    "NIA_SL_SEN0002_REAL17_F"
]

data_dir = Path("data/keypoint/17")

WIDTH = 1920
HEIGHT = 1080

for video_id in video_ids:

    bodies, faces, left_hands, right_hands = load_video_keypoints(
        data_dir, video_id
    )

    print(f"\n=== {video_id} ===")
    print("전체 프레임:", len(bodies))

    for name, points in [
        ("Body", bodies),
        ("Face", faces),
        ("Left Hand", left_hands),
        ("Right Hand", right_hands)
    ]:
        x = points[:, :, 0]
        y = points[:, :, 1]
        confidence = points[:, :, 2]

        # 결측치 및 비정상 값 검사
        zero_xy = (x == 0) & (y == 0)
        nan_mask = np.isnan(points).any(axis=2)
        inf_mask = np.isinf(points).any(axis=2)

        out_of_bounds = (
            (x < 0) | (x >= WIDTH) |
            (y < 0) | (y >= HEIGHT)
        )

        zero_conf = confidence == 0

        low_conf = (
            (confidence > 0) & (confidence < 0.5)
        )

        print(f"\n{name}")
        print("전체 키포인트:", x.size)
        print("(0, 0) 좌표:", np.count_nonzero(zero_xy))
        print("NaN:", np.count_nonzero(nan_mask))
        print("Inf:", np.count_nonzero(inf_mask))
        print("이미지 범위 초과:", np.count_nonzero(out_of_bounds))
        print("Confidence 0:", np.count_nonzero(zero_conf))
        print("낮은 Confidence:", np.count_nonzero(low_conf))