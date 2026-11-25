from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class SignLanguageDataset(Dataset):
    def __init__(self, data_dir, mode="manual"):
        self.files = sorted(Path(data_dir).glob("*.npz"))
        self.mode = mode

        if mode not in ("manual", "manual_face"):
            raise ValueError("mode must be 'manual' or 'manual_face'")

        if not self.files:
            raise FileNotFoundError("NPZ 파일을 찾을 수 없습니다.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        with np.load(file_path) as data:
            keypoints = data[self.mode].copy()

        return torch.from_numpy(keypoints).float()

if __name__ == "__main__":
    dataset = SignLanguageDataset("outputs", mode="manual")

    print("영상 개수:", len(dataset))

    sample = dataset[0]

    print("데이터 타입:", type(sample))
    print("Tensor shape:", sample.shape)
    print("Tensor dtype:", sample.dtype)
