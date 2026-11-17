"""GRU/1D CNN 수어 어휘 분류 모델 인터페이스 (미구현)."""


def build_model(name="gru", *, num_classes, input_size, hidden_size=128,
                num_layers=2, dropout=0.2):
    """(B,T,J,C)를 받아 (B,num_classes) logits을 출력할 모델을 구성한다.

    input_size는 확정된 J*C이며 클래스 수와 함께 실제 데이터로 결정한다.
    """
    # TODO: GRU 또는 1D CNN 및 분류층 구현
    raise NotImplementedError("수어 시계열 분류 모델을 구현해야 합니다.")

#dkdkdkkdkdkdk