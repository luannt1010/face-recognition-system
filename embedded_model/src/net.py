import torch.nn as nn
try:
    from src.iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
except ModuleNotFoundError:
    from iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200

class ResNetEncoder(nn.Module):
    def __init__(self, model_size=18, embedding_dim=512, dropout=0.4):
        super().__init__()
        self.model_size = model_size
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.backbone = self._create_model()

    def _create_model(self):

        if self.model_size not in [18, 34, 50, 100, 200]:
            raise ValueError("Model size is invalid! Must in [18, 34, 50, 100, 200]")
        if self.model_size == 50:
            backbone = iresnet50(num_features=self.embedding_dim, dropout=self.dropout)
        elif self.model_size == 18:
            backbone = iresnet18(num_features=self.embedding_dim, dropout=self.dropout)
        elif self.model_size == 34:
            backbone = iresnet34(num_features=self.embedding_dim, dropout=self.dropout)
        elif self.model_size == 100:
            backbone = iresnet100(num_features=self.embedding_dim, dropout=self.dropout)
        elif self.model_size == 200:
            backbone = iresnet200(num_features=self.embedding_dim, dropout=self.dropout)
        return backbone

    def forward(self, x):
        x = self.backbone(x)
        return x



