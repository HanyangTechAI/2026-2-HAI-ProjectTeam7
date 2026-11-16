"""Manifest 기반 수어 키포인트 Dataset 인터페이스 (미구현)."""

from torch.utils.data import Dataset


class SignLanguageDataset(Dataset):
    """manual/fusion (T,J,C), label을 반환할 예정인 Dataset."""

    def __init__(self, manifest_path, split, transform=None):
        self.manifest_path = manifest_path
        self.split = split
        self.transform = transform
        # TODO: class_map과 split_manifest 연결, 스키마 및 파일 경로 검증
        raise NotImplementedError("수어 Manifest 및 Tensor 로딩을 구현해야 합니다.")

    def __len__(self):
        raise NotImplementedError("샘플 수 반환을 구현해야 합니다.")

    def __getitem__(self, index):
        # TODO: 전처리 Tensor 로딩; 두 입력에 동기화된 transform 적용
        raise NotImplementedError("manual/fusion/label 반환을 구현해야 합니다.")


def build_dataloader(dataset, batch_size=32, shuffle=False, num_workers=0):
    """학습 split에만 shuffle을 적용하는 DataLoader 구성 예정."""
    raise NotImplementedError("DataLoader 구성을 구현해야 합니다.")
