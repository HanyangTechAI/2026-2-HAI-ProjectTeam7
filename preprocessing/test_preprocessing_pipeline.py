import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from missing import handle_missing_keypoints

from preprocess import (
    load_video_keypoints,
    normalize_keypoints,
    make_model_input,
    make_model_mask,
    save_npz
)


def test_pipeline(video_id):
    data_dir = Path("data/keypoint/17")

    bodies, faces, left_hands, right_hands = (
        load_video_keypoints(data_dir, video_id)
    )

    T = len(bodies)

    print(f"\n=== {video_id} ===")

    # 1. 인위적인 결측치 생성
    left_hands[10:12] = 0
    faces[20:23] = 0
    right_hands[30:35] = 0

    bodies[40:44, [2, 5], :] = 0
    bodies[60:66, [2, 5], :] = 0

    # 2. 키포인트 결측치 처리
    bodies, body_original, body_valid = (
        handle_missing_keypoints(bodies, max_gap=3)
    )

    faces, face_original, face_valid = (
        handle_missing_keypoints(faces, max_gap=3)
    )

    left_hands, left_original, left_valid = (
        handle_missing_keypoints(left_hands, max_gap=3)
    )

    right_hands, right_original, right_valid = (
        handle_missing_keypoints(right_hands, max_gap=3)
    )

    # 3. 정규화
    (
        bodies, faces, left_hands,
        right_hands, reference_valid
    ) = normalize_keypoints(
        bodies, faces, left_hands, right_hands,
        body_valid, face_valid, left_valid, right_valid
    )

    # 4. 최종 Mask 갱신
    body_valid &= reference_valid[:, None]
    face_valid &= reference_valid[:, None]
    left_valid &= reference_valid[:, None]
    right_valid &= reference_valid[:, None]

    # 5. 개별 결측 상황 검증

    # 왼손 2프레임 보간
    assert not np.any(left_original[10:12])
    assert np.all(left_valid[10:12])
    assert np.all(left_hands[10:12, :, 2] == 0)

    print("Left Hand Interpolation: Passed")

    # 얼굴 3프레임 보간
    assert not np.any(face_original[20:23])
    assert np.all(face_valid[20:23])
    assert np.all(faces[20:23, :, 2] == 0)

    print("Face Interpolation: Passed")

    # 오른손 5프레임 Mask 처리
    assert not np.any(right_original[30:35])
    assert not np.any(right_valid[30:35])
    assert np.all(right_hands[30:35] == 0)

    print("Right Hand Masking: Passed")

    # 양쪽 어깨 4프레임 결측
    assert np.all(reference_valid[40:44])

    assert not np.any(body_valid[40:44, [2, 5]])
    assert np.all(bodies[40:44, [2, 5], :] == 0)

    assert np.all(left_valid[40:44])
    assert np.all(right_valid[40:44])
    assert np.all(face_valid[40:44])

    print("Reference Repair: Passed")

    # 양쪽 어깨 6프레임 결측
    assert not np.any(reference_valid[60:66])

    assert not np.any(body_valid[60:66])
    assert not np.any(face_valid[60:66])
    assert not np.any(left_valid[60:66])
    assert not np.any(right_valid[60:66])

    assert np.all(bodies[60:66] == 0)
    assert np.all(faces[60:66] == 0)
    assert np.all(left_hands[60:66] == 0)
    assert np.all(right_hands[60:66] == 0)

    print("Invalid Reference Masking: Passed")

    # 6. Manual / Holistic 생성
    manual = make_model_input(
        bodies, faces, left_hands, right_hands,
        mode="manual"
    )

    manual_face = make_model_input(
        bodies, faces, left_hands, right_hands,
        mode="manual_face"
    )

    manual_original = make_model_mask(
        body_original, face_original,
        left_original, right_original,
        mode="manual"
    )

    holistic_original = make_model_mask(
        body_original, face_original,
        left_original, right_original,
        mode="manual_face"
    )

    manual_valid = make_model_mask(
        body_valid, face_valid,
        left_valid, right_valid,
        mode="manual"
    )

    holistic_valid = make_model_mask(
        body_valid, face_valid,
        left_valid, right_valid,
        mode="manual_face"
    )

    # 7. Shape 및 일관성 검증
    assert manual.shape == (T, 67, 3)
    assert manual_face.shape == (T, 137, 3)

    assert manual_valid.shape == (T, 67)
    assert holistic_valid.shape == (T, 137)

    assert np.array_equal(
        manual, manual_face[:, :67]
    )

    assert np.array_equal(
        manual_original, holistic_original[:, :67]
    )

    assert np.array_equal(
        manual_valid, holistic_valid[:, :67]
    )

    assert np.all(manual[~manual_valid] == 0)
    assert np.all(manual_face[~holistic_valid] == 0)

    assert np.isfinite(manual).all()
    assert np.isfinite(manual_face).all()

    print("Shape and Consistency: Passed")

    # 8. 임시 NPZ 저장 및 다시 불러오기
    with TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test_pipeline.npz"

        save_npz(
            output_path,
            manual,
            manual_face,
            manual_original,
            holistic_original,
            manual_valid,
            holistic_valid,
            reference_valid
        )

        expected = {
            "manual": manual,
            "manual_face": manual_face,
            "manual_original_mask": manual_original,
            "manual_face_original_mask": holistic_original,
            "manual_valid_mask": manual_valid,
            "manual_face_valid_mask": holistic_valid,
            "reference_valid": reference_valid
        }

        with np.load(output_path) as data:
            assert set(data.files) == set(expected)

            for key, value in expected.items():
                assert np.array_equal(data[key], value)

    print("NPZ Save and Load: Passed")

    # 9. 최종 통계
    normal = holistic_original & holistic_valid
    interpolated = ~holistic_original & holistic_valid
    invalid = ~holistic_valid

    print("\nTotal Keypoints:", holistic_valid.size)
    print("Normal Keypoints:", np.sum(normal))
    print("Interpolated Keypoints:", np.sum(interpolated))
    print("Invalid Keypoints:", np.sum(invalid))
    print("Valid Reference Frames:", np.sum(reference_valid))


if __name__ == "__main__":
    video_ids = [
        "NIA_SL_SEN0001_REAL17_F",
        "NIA_SL_SEN0002_REAL17_F"
    ]

    for video_id in video_ids:
        test_pipeline(video_id)

    print("\nAll preprocessing pipeline tests passed!")