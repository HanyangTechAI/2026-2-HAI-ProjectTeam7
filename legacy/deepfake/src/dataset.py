"""Manifest 기반 Real/Fake Dataset 템플릿."""

from torch.utils.data import Dataset


class DeepfakeDataset(Dataset):
    """CSV의 이미지 경로와 이진 라벨을 읽는 Dataset."""

    def __init__(self, manifest_path, transform=None):
        self.manifest_path = manifest_path
        self.transform = transform
        # TODO: manifest 로딩 및 필수 컬럼 검증
        raise NotImplementedError("Manifest 로딩을 구현해야 합니다.")

    def __len__(self):
        raise NotImplementedError("샘플 수 반환을 구현해야 합니다.")

    def __getitem__(self, index):
        # TODO: 이미지 로딩, transform 적용, 이미지와 라벨 반환
        raise NotImplementedError("샘플 로딩을 구현해야 합니다.")
