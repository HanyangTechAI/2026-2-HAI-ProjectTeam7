import numpy as np
from pathlib import Path

video_ids = [
    "NIA_SL_SEN0001_REAL17_F",
    "NIA_SL_SEN0002_REAL17_F"
]

output_dir = Path("outputs")

for video_id in video_ids:
    print(f"\n=== {video_id} ===")

    with np.load(output_dir / f"{video_id}.npz") as data:

        # 저장된 항목 및 Shape 확인
        for key in data.files:
            print(
                f"{key}: {data[key].shape}, "
                f"dtype={data[key].dtype}"
            )

        manual = data["manual"]
        manual_face = data["manual_face"]

        manual_original = data["manual_original_mask"]
        holistic_original = data["manual_face_original_mask"]

        manual_valid = data["manual_valid_mask"]
        holistic_valid = data["manual_face_valid_mask"]

        reference_valid = data["reference_valid"]

        T = manual.shape[0]

        # 1. Shape 검증
        assert manual.shape == (T, 67, 3)
        assert manual_face.shape == (T, 137, 3)

        assert manual_original.shape == (T, 67)
        assert holistic_original.shape == (T, 137)

        assert manual_valid.shape == (T, 67)
        assert holistic_valid.shape == (T, 137)

        assert reference_valid.shape == (T,)

        print("\nShape Test: Passed")

        # 2. Mask 자료형 검증
        for mask in [
            manual_original,
            holistic_original,
            manual_valid,
            holistic_valid,
            reference_valid
        ]:
            assert mask.dtype == np.bool_

        print("Mask Type Test: Passed")

        # 3. Manual과 Holistic 일치 여부
        assert np.array_equal(
            manual, manual_face[:, :67]
        )

        assert np.array_equal(
            manual_original, holistic_original[:, :67]
        )

        assert np.array_equal(
            manual_valid, holistic_valid[:, :67]
        )

        print("Manual-Holistic Consistency: Passed")

        # 4. 유효하지 않은 좌표 검증
        assert np.all(manual[~manual_valid] == 0)
        assert np.all(manual_face[~holistic_valid] == 0)

        assert not np.any(manual_valid[~reference_valid])
        assert not np.any(holistic_valid[~reference_valid])

        print("Invalid Keypoint Test: Passed")

        # 5. 정상, 보간, 사용 불가 키포인트 개수
        normal = holistic_original & holistic_valid
        interpolated = ~holistic_original & holistic_valid
        invalid = ~holistic_valid

        print("\nTotal Keypoints:", holistic_valid.size)
        print("Normal Keypoints:", np.sum(normal))
        print("Interpolated Keypoints:", np.sum(interpolated))
        print("Invalid Keypoints:", np.sum(invalid))
        print("Valid Reference Frames:", np.sum(reference_valid))

print("\nAll mask tests passed!")