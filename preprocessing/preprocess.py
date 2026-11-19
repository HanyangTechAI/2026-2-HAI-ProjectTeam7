import json
import numpy as np
from pathlib import Path


def load_keypoints(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    person = data["people"]

    body = np.array(person["pose_keypoints_2d"]).reshape(25, 3)
    face = np.array(person["face_keypoints_2d"]).reshape(70, 3)
    left_hand = np.array(person["hand_left_keypoints_2d"]).reshape(21, 3)
    right_hand = np.array(person["hand_right_keypoints_2d"]).reshape(21, 3)

    return body, face, left_hand, right_hand

def load_video_keypoints(folder_path, video_id):
    folder = Path(folder_path)

    json_files = sorted(
        folder.rglob(f"{video_id}_*_keypoints.json")
    )

    if not json_files:
        raise FileNotFoundError(
            f"No keypoint files found for {video_id}"
        )

    bodies = []
    faces = []
    left_hands = []
    right_hands = []

    for json_path in json_files:
        body, face, left_hand, right_hand = load_keypoints(json_path)

        bodies.append(body)
        faces.append(face)
        left_hands.append(left_hand)
        right_hands.append(right_hand)

    bodies = np.stack(bodies)
    faces = np.stack(faces)
    left_hands = np.stack(left_hands)
    right_hands = np.stack(right_hands)

    return bodies, faces, left_hands, right_hands

def make_model_input(bodies, faces, left_hands, right_hands, mode="manual"):
    if mode == "manual":
        result = np.concatenate(
            [bodies, left_hands, right_hands],
            axis=1
        )

    elif mode == "manual_face":
        result = np.concatenate(
            [bodies, left_hands, right_hands, faces],
            axis=1
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return result

def save_npz(output_path, manual, manual_face):
    np.savez_compressed(
        output_path,
        manual=manual,
        manual_face=manual_face
    )

if __name__ == "__main__":
    video_ids = [
        "NIA_SL_SEN0001_REAL17_F",
        "NIA_SL_SEN0002_REAL17_F"
    ]

    data_dir = Path("data/keypoint/17")
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    for video_id in video_ids:
        print(f"Processing: {video_id}")

        bodies, faces, left_hands, right_hands = load_video_keypoints(
            data_dir, video_id
        )

        manual = make_model_input(
            bodies, faces, left_hands, right_hands,
            mode="manual"
        )

        manual_face = make_model_input(
            bodies, faces, left_hands, right_hands,
            mode="manual_face"
        )

        output_path = output_dir / f"{video_id}.npz"

        save_npz(output_path, manual, manual_face)

        print("Manual-only:", manual.shape)
        print("Manual+Face:", manual_face.shape)
        print(f"Saved: {output_path}")
