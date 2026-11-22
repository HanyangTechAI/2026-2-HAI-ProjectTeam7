import numpy as np
from pathlib import Path

output_dir = Path("outputs")

video_ids = [
    "NIA_SL_SEN0001_REAL17_F",
    "NIA_SL_SEN0002_REAL17_F"
]

for video_id in video_ids:
    file_path = output_dir / f"{video_id}.npz"

    with np.load(file_path) as data:
        manual = data["manual"]
        manual_face = data["manual_face"]

    # BODY_25: 오른쪽 어깨 2, 왼쪽 어깨 5
    right_shoulder = manual[:, 2, :2]
    left_shoulder = manual[:, 5, :2]

    # 정규화 후 어깨 중심
    center = (right_shoulder + left_shoulder) / 2

    # 정규화 후 어깨 너비
    shoulder_width = np.linalg.norm(
        right_shoulder - left_shoulder,
        axis=1
    )

    # 전체 프레임 검사
    center_ok = np.allclose(center, 0, atol=1e-5)
    width_ok = np.allclose(shoulder_width, 1, atol=1e-5)

    # Manual과 Manual+Face의 공통 키포인트 비교
    same_manual = np.allclose(
        manual,
        manual_face[:, :67, :],
        atol=1e-6
    )

    # Confidence가 변경되지 않았는지 확인
    confidence_ok = np.all(
        manual_face[:, :, 2] == 1.0
    )

    print(f"\n=== {video_id} ===")
    print("전체 프레임:", len(manual))
    print("어깨 중심 (0, 0):", center_ok)
    print("어깨 너비 1:", width_ok)
    print("Manual 좌표 일치:", same_manual)
    print("Confidence 유지:", confidence_ok)