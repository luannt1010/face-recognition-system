import torch.nn as nn
try:
    from src.iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
except ModuleNotFoundError:
    from iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200

class IResNetEncoder(nn.Module):
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

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_connection=False):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skip_connection = skip_connection
        if self.skip_connection:
            self.shortcut = self.make_shortcut(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
    
    def make_shortcut(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.BatchNorm2d(out_channels))
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.skip_connection:
            x += self.shortcut(identity)
        x = self.relu(x)
        x = self.max_pool(x)
        return x 


class MyModel(nn.Module):
    def __init__(self, embedding_dim=512, dropout=0.4):
        super().__init__()
        self.block1 = CNNBlock(in_channels=3, out_channels=32, skip_connection=True)
        self.block2 = CNNBlock(in_channels=32, out_channels=64, skip_connection=True)
        self.block3 = CNNBlock(in_channels=64, out_channels=128, skip_connection=True)
        self.block4 = CNNBlock(in_channels=128, out_channels=256, skip_connection=True)
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.extract_embed = nn.Sequential(nn.Flatten(), 
                                        nn.Linear(256*7*7, 1024), nn.ReLU(), nn.Dropout(self.dropout), 
                                        nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(self.dropout), 
                                        nn.Linear(512, self.embedding_dim))
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.extract_embed(x)
        return x



