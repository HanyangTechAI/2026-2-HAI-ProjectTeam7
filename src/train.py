"""GRU 어휘 분류 모델의 학습 및 검증 스크립트."""

import argparse
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def make_dummy_loaders(sequence_length, input_size, num_classes, batch_size, sample_count):
    ###실제 Dataset 연결 전 학습 루프를 확인할 랜덤 데이터를 만듦
    validation_count = max(1, sample_count // 4)
    train_inputs = torch.randn(sample_count, sequence_length, input_size)
    train_labels = torch.randint(num_classes, (sample_count,))
    validation_inputs = torch.randn(validation_count, sequence_length, input_size)
    validation_labels = torch.randint(num_classes, (validation_count,))

    train_loader = DataLoader(
        TensorDataset(train_inputs, train_labels),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(validation_inputs, validation_labels),
        batch_size=batch_size,
    )
    return train_loader, validation_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="학습 설정 YAML 경로",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="실제 데이터 대신 랜덤 데이터로 학습/검증 루프를 시험",
    )
    parser.add_argument("--input-size", type=int, default=126, help="입력 특징 수 F")
    parser.add_argument("--num-classes", type=int, default=10, help="분류할 Gloss 클래스 수")
    parser.add_argument("--samples", type=int, default=64, help="더미 학습 샘플 수")
    parser.add_argument("--epochs", type=int, help="설정 파일의 epoch 수 대신 사용할 값")
    args = parser.parse_args()

    if not args.dummy:
        parser.error(
            "실제 Dataset은 아직 구현되지 않았습니다. "
            "학습 루프 테스트에는 --dummy 옵션을 사용하세요."
        )

    config_path = Path(args.config)
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        parser.error(f"설정 파일이 비어 있거나 올바른 YAML이 아닙니다: {config_path}")

    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_config = config.get("model", {})
    train_config = config.get("train", {})
    data_config = config.get("data", {})
    num_classes = model_config.get("num_classes") or args.num_classes
    sequence_length = data_config.get("sequence_length") or 32
    batch_size = train_config.get("batch_size") or 32
    epochs = args.epochs or train_config.get("epochs") or 1

    device_setting = train_config.get("device", "auto")
    if device_setting == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_setting)
        if device.type == "cuda" and not torch.cuda.is_available():
            parser.error("설정에서 CUDA를 요청했지만 사용할 수 없습니다.")

    train_loader, validation_loader = make_dummy_loaders(
        sequence_length=sequence_length,
        input_size=args.input_size,
        num_classes=num_classes,
        batch_size=batch_size,
        sample_count=args.samples,
    )
    model = build_model(
        name=model_config.get("name", "gru"),
        num_classes=num_classes,
        input_size=args.input_size,
        hidden_size=model_config.get("hidden_size", 128),
        num_layers=model_config.get("num_layers", 2),
        dropout=model_config.get("dropout", 0.2),
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.get("learning_rate", 0.001),
        weight_decay=train_config.get("weight_decay", 0.0001),
    )

    print(
        f"더미 데이터 학습 시작: device={device}, "
        f"input=(T={sequence_length}, F={args.input_size}), "
        f"classes={num_classes}, epochs={epochs}"
    )
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, criterion, device
        )
        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss={train_loss:.4f}, accuracy={train_accuracy:.3f} | "
            f"validation loss={validation_loss:.4f}, "
            f"accuracy={validation_accuracy:.3f}"
        )

    print("더미 데이터 Forward/Backward 및 학습/검증 루프 테스트 완료")


if __name__ == "__main__":
    main()
