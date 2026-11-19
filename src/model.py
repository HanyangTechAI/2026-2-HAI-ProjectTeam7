#"""GRU/1D CNN 수어 어휘 분류 모델 인터페이스 (미구현)."""
import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout,
    ):
        super().__init__()

        self.input_size = input_size

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True, ##실시간 한프레임씩 들어오면 False로 바꿀필요있음
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        ## 양방향이므로 정방향과 역방향의 hidden state를 합치기
        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes,
        )

    def forward(self, x):
        """
        입력:
            x: (B, T, J, C)

        출력:
            logits: (B, num_classes)
        """

        if x.ndim != 4:
            raise ValueError(
                f"입력은 (B, T, J, C) 형태여야 합니다. "
                f"현재 형태: {tuple(x.shape)}"
            )

        batch_size, num_frames, num_joints, num_coords = x.shape

        if num_joints * num_coords != self.input_size:
            raise ValueError(
                f"J*C는 input_size와 같아야 합니다. "
                f"J*C={num_joints * num_coords}, "
                f"input_size={self.input_size}"
            )

        # (B, T, J, C) → (B, T, J*C)
        x = x.reshape(batch_size, num_frames, self.input_size)

        # h_n: (num_layers * 2, B, hidden_size)
        _, h_n = self.gru(x)

        # 마지막 층의 정방향, 역방향 hidden state
        forward_hidden = h_n[-2]
        backward_hidden = h_n[-1]

        # (B, hidden_size * 2)
        combined = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1,
        )

        combined = self.dropout(combined)

        # (B, num_classes)
        logits = self.classifier(combined)

        return logits


def build_model(
    name="gru",
    *,
    num_classes,
    input_size,
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
):
    """(B, T, J, C)를 받아 (B, num_classes) logits을 출력하는 모델."""

    if name.lower() == "gru":
        return GRUClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

    raise ValueError(
        f"지원하지 않는 모델: {name!r}. 현재는 'gru'만 지원합니다."
    )