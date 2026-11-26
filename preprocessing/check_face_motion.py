import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from preprocess import load_video_keypoints


video_ids = [
    "NIA_SL_SEN0001_REAL17_F",
    "NIA_SL_SEN0002_REAL17_F"
]

data_dir = Path("data/keypoint/17")
output_dir = Path("outputs")


def point_distance(points, index1, index2):
    return np.linalg.norm(
        points[:, index1, :2]
        - points[:, index2, :2],
        axis=1
    )


def get_face_features(faces):

    # 입을 벌리는 정도
    mouth = (
        point_distance(faces, 61, 67)
        + point_distance(faces, 62, 66)
        + point_distance(faces, 63, 65)
    ) / 3

    # 눈을 뜨고 감는 정도
    right_eye = (
        point_distance(faces, 37, 41)
        + point_distance(faces, 38, 40)
    ) / 2

    left_eye = (
        point_distance(faces, 43, 47)
        + point_distance(faces, 44, 46)
    ) / 2

    eye = (right_eye + left_eye) / 2

    # 눈썹과 눈 사이의 거리
    right_eyebrow = point_distance(faces, 19, 37)
    left_eyebrow = point_distance(faces, 24, 44)

    eyebrow = (right_eyebrow + left_eyebrow) / 2

    return {
        "Mouth Opening": mouth,
        "Eye Opening": eye,
        "Eyebrow Movement": eyebrow
    }


def standardize(values):
    std = np.std(values)

    if std < 1e-8:
        return np.zeros_like(values)

    return (values - np.mean(values)) / std


def check_face_motion(video_id):

    bodies, raw_faces, left_hands, right_hands = (
        load_video_keypoints(data_dir, video_id)
    )

    # 정규화된 얼굴 좌표 불러오기
    with np.load(output_dir / f"{video_id}.npz") as data:
        normalized_faces = data["manual_face"][:, 67:, :]
        face_valid = data["manual_face_valid_mask"][:, 67:]

    assert raw_faces.shape == normalized_faces.shape

    # 현재 검증은 모든 얼굴 좌표가 유효한 샘플을 대상으로 함
    assert np.all(face_valid)

    raw_features = get_face_features(raw_faces)
    normalized_features = get_face_features(normalized_faces)

    print(f"\n=== {video_id} ===")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    for ax, name in zip(axes, raw_features):

        raw = raw_features[name]
        normalized = normalized_features[name]

        raw_standardized = standardize(raw)
        normalized_standardized = standardize(normalized)

        # 움직임 패턴의 상관계수
        if np.std(raw) > 1e-8 and np.std(normalized) > 1e-8:
            correlation = np.corrcoef(raw, normalized)[0, 1]
        else:
            correlation = np.nan

        print(f"\n{name}")
        print(f"Raw Mean: {np.mean(raw):.6f}")
        print(f"Normalized Mean: {np.mean(normalized):.6f}")
        print(f"Correlation: {correlation:.6f}")

        ax.plot(
            raw_standardized,
            label="Raw",
            linestyle="--"
        )

        ax.plot(
            normalized_standardized,
            label="Normalized",
            alpha=0.8
        )

        ax.set_title(f"{name} | Correlation: {correlation:.4f}")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Standardized Value")
        ax.legend()
        ax.grid(True)

    fig.suptitle(video_id)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for video_id in video_ids:
        check_face_motion(video_id)