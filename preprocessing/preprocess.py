import json
import numpy as np
from pathlib import Path
from missing import handle_missing_keypoints

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

def repair_normalization_reference(
    center, scale, shoulder_valid, max_gap=4
):
    center = center.copy()
    scale = scale.copy()
    reference_valid = shoulder_valid.copy()

    T = len(scale)
    t = 0

    while t < T:
        # 정상적인 정규화 기준은 그대로 유지
        if shoulder_valid[t]:
            t += 1
            continue

        # 연속된 결측 구간 찾기
        start = t

        while t < T and not shoulder_valid[t]:
            t += 1

        end = t
        gap = end - start

        # 최대 허용 길이를 초과하면 복원하지 않음
        if gap > max_gap:
            continue

        # 중간 결측: 앞뒤 정상 프레임을 이용한 선형 보간
        if start > 0 and end < T:
            for i in range(gap):
                ratio = (i + 1) / (gap + 1)

                center[start + i] = (
                    center[start - 1]
                    + ratio * (center[end] - center[start - 1])
                )

                scale[start + i] = (
                    scale[start - 1]
                    + ratio * (scale[end] - scale[start - 1])
                )

            reference_valid[start:end] = True

        # 시작 부분 결측: 이후 정상 프레임 활용
        elif start == 0 and end < T:
            center[start:end] = center[end]
            scale[start:end] = scale[end]
            reference_valid[start:end] = True

        # 끝부분 결측: 이전 정상 프레임 활용
        elif start > 0 and end == T:
            center[start:end] = center[start - 1]
            scale[start:end] = scale[start - 1]
            reference_valid[start:end] = True

    return center, scale, reference_valid

def normalize_keypoints(
    bodies, faces, left_hands, right_hands,
    body_valid_mask=None,
    face_valid_mask=None,
    left_valid_mask=None,
    right_valid_mask=None
):
    # 원본 데이터가 변경되지 않도록 복사
    bodies = bodies.astype(float).copy()
    faces = faces.astype(float).copy()
    left_hands = left_hands.astype(float).copy()
    right_hands = right_hands.astype(float).copy()

    # 오른쪽 어깨: 2, 왼쪽 어깨: 5
    right_shoulder = bodies[:, 2, :2]
    left_shoulder = bodies[:, 5, :2]

    # 어깨 중심과 너비 계산
    center = (right_shoulder + left_shoulder) / 2

    scale = np.linalg.norm(
        right_shoulder - left_shoulder,
        axis=1
    )

    # 정규화 기준의 유효성 확인
    shoulder_valid = (
        np.isfinite(center).all(axis=1)
        & np.isfinite(scale)
        & (scale > 1e-6)
    )

    if body_valid_mask is not None:
        shoulder_valid &= (
            body_valid_mask[:, 2]
            & body_valid_mask[:, 5]
        )

    # 정규화 기준값 결측 처리
    center, scale, reference_valid = \
        repair_normalization_reference(
            center, scale, shoulder_valid,
            max_gap=4
        )

    # 정규화 가능한 프레임이 하나도 없는 경우
    if not np.any(reference_valid):
        raise ValueError("정규화 가능한 프레임이 없습니다.")

    # 복원 불가능한 기준값의 연산 오류 방지
    center[~reference_valid] = 0
    scale[~reference_valid] = 1

    # 신체 부위별 정규화
    for points, mask in [
        (bodies, body_valid_mask),
        (faces, face_valid_mask),
        (left_hands, left_valid_mask),
        (right_hands, right_valid_mask)
    ]:
        if mask is None:
            mask = np.ones(points.shape[:2], dtype=bool)

        # 키포인트와 정규화 기준이 모두 유효해야 함
        effective_mask = mask & reference_valid[:, None]

        normalized_xy = (
            points[:, :, :2] - center[:, None, :]
        ) / scale[:, None, None]

        # 유효한 좌표만 정규화
        points[effective_mask, :2] = normalized_xy[effective_mask]

        # 유효하지 않은 좌표는 0으로 처리
        points[~effective_mask] = 0

    return bodies, faces, left_hands, right_hands, reference_valid

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

def make_model_mask(
    body_mask, face_mask, left_mask, right_mask,
    mode="manual"
):
    if mode == "manual":
        result = np.concatenate(
            [body_mask, left_mask, right_mask],
            axis=1
        )

    elif mode == "manual_face":
        result = np.concatenate(
            [body_mask, left_mask, right_mask, face_mask],
            axis=1
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return result

def save_npz(
    output_path,
    manual,
    manual_face,
    manual_original_mask,
    manual_face_original_mask,
    manual_valid_mask,
    manual_face_valid_mask,
    reference_valid
):
    np.savez_compressed(
        output_path,
        manual=manual,
        manual_face=manual_face,
        manual_original_mask=manual_original_mask,
        manual_face_original_mask=manual_face_original_mask,
        manual_valid_mask=manual_valid_mask,
        manual_face_valid_mask=manual_face_valid_mask,
        reference_valid=reference_valid
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
        # 결측치 처리
        bodies, body_original_mask, body_valid_mask = \
            handle_missing_keypoints(bodies, max_gap=3)

        faces, face_original_mask, face_valid_mask = \
            handle_missing_keypoints(faces, max_gap=3)

        left_hands, left_original_mask, left_valid_mask = \
            handle_missing_keypoints(left_hands, max_gap=3)

        right_hands, right_original_mask, right_valid_mask = \
            handle_missing_keypoints(right_hands, max_gap=3)

        bodies, faces, left_hands, right_hands, reference_valid = normalize_keypoints(
            bodies, faces, left_hands, right_hands,
            body_valid_mask,
            face_valid_mask,
            left_valid_mask,
            right_valid_mask
        )

        # 정규화 기준의 유효성을 키포인트 Mask에 반영
        body_valid_mask &= reference_valid[:, None]
        face_valid_mask &= reference_valid[:, None]
        left_valid_mask &= reference_valid[:, None]
        right_valid_mask &= reference_valid[:, None]

        manual = make_model_input(
            bodies, faces, left_hands, right_hands,
            mode="manual"
        )

        manual_face = make_model_input(
            bodies, faces, left_hands, right_hands,
            mode="manual_face"
        )

        manual_original_mask = make_model_mask(
            body_original_mask,
            face_original_mask,
            left_original_mask,
            right_original_mask,
            mode="manual"
        )

        manual_face_original_mask = make_model_mask(
            body_original_mask,
            face_original_mask,
            left_original_mask,
            right_original_mask,
            mode="manual_face"
        )

        manual_valid_mask = make_model_mask(
            body_valid_mask,
            face_valid_mask,
            left_valid_mask,
            right_valid_mask,
            mode="manual"
        )

        manual_face_valid_mask = make_model_mask(
            body_valid_mask,
            face_valid_mask,
            left_valid_mask,
            right_valid_mask,
            mode="manual_face"
        )

        output_path = output_dir / f"{video_id}.npz"

        save_npz(
            output_path,
            manual,
            manual_face,
            manual_original_mask,
            manual_face_original_mask,
            manual_valid_mask,
            manual_face_valid_mask,
            reference_valid
        )

        print("Manual-only:", manual.shape)
        print("Manual+Face:", manual_face.shape)
        print(f"Saved: {output_path}")
