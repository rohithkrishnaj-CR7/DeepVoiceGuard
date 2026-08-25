"""
Deep Learning Architectures for Audio Anti-Spoofing and Deepfake Detection.
Includes:
- LCNN (Lightweight CNN with Max-Feature-Map activation for LFCCs, ASVspoof benchmark)
- SpecResNet (Residual Network for Log-Mel Spectrograms with Temporal Attention)
- BiLSTMAcoustic (Bi-directional LSTM for sequential acoustic frames)
- DeepClassifierWrapper (Unified training and inference engine)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Optional, Dict, Any

class MaxFeatureMap2D(nn.Module):
    """
    Max-Feature-Map (MFM) 2D activation layer.
    Splits channel dimensions into two halves and computes element-wise maximum.
    Widely recognized in ASVspoof research for separating genuine speech from synthetic artifacts.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out_a, out_b = torch.chunk(out, 2, dim=1)
        return torch.max(out_a, out_b)

class MaxFeatureMapLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, 2 * out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        out_a, out_b = torch.chunk(out, 2, dim=1)
        return torch.max(out_a, out_b)

class LCNN(nn.Module):
    """
    Lightweight Convolutional Neural Network with Max-Feature-Map activation (LFCC input).
    Input shape: (Batch, 1, 60, T_frames)
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.layer1 = nn.Sequential(
            MaxFeatureMap2D(in_channels, 16, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16)
        )
        self.layer2 = nn.Sequential(
            MaxFeatureMap2D(16, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            MaxFeatureMap2D(24, 24, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(24)
        )
        self.layer3 = nn.Sequential(
            MaxFeatureMap2D(24, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            MaxFeatureMap2D(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32)
        )
        self.layer4 = nn.Sequential(
            MaxFeatureMap2D(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc_block = nn.Sequential(
            MaxFeatureMapLinear(48 * 4 * 4, 128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, 60, T)
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = torch.flatten(h, 1)
        out = self.fc_block(h)
        return out

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

class SpecResNet(nn.Module):
    """
    2D Residual Network with Squeeze-and-Excitation / Attention pooling for Log-Mel Spectrograms.
    Input shape: (Batch, 1, 128, T_frames)
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.block1 = ResBlock(32)
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block2 = ResBlock(64)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.block3 = ResBlock(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        h = self.init_conv(x)
        h = self.block1(h)
        h = self.down1(h)
        h = self.block2(h)
        h = self.down2(h)
        h = self.block3(h)
        h = self.pool(h)
        h = torch.flatten(h, 1)
        return self.classifier(h)

class BiLSTMAcoustic(nn.Module):
    """
    Bidirectional LSTM network for sequential acoustic frames.
    """
    def __init__(self, input_dim: int = 60, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, input_dim, T) -> transpose to (B, T, input_dim)
        if x.ndim == 3 and x.shape[1] < x.shape[2]:
            x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        # Average pooling over time
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(pooled)

class DeepClassifierWrapper:
    """
    Unified trainer and inference engine for PyTorch anti-spoofing models.
    """
    def __init__(self, model_type: str = 'lcnn', lr: float = 1e-3, device: Optional[str] = None):
        self.model_type = model_type.lower()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if self.model_type == 'lcnn':
            self.model = LCNN().to(self.device)
        elif self.model_type == 'specresnet':
            self.model = SpecResNet().to(self.device)
        elif self.model_type == 'bilstm':
            self.model = BiLSTMAcoustic().to(self.device)
        else:
            raise ValueError(f"Unknown deep model type: {model_type}")

        self.lr = lr
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 15,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Trains model using CrossEntropyLoss and Adam optimizer with CosineAnnealing LR.
        """
        self.model.train()
        t_X = torch.tensor(X_train, dtype=torch.float32)
        t_y = torch.tensor(y_train, dtype=torch.long)
        dataset = TensorDataset(t_X, t_y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(batch_y)

            train_loss = running_loss / len(dataset)
            history['train_loss'].append(train_loss)
            scheduler.step()

            if X_val is not None and y_val is not None:
                val_loss = self.evaluate_loss(X_val, y_val, criterion)
                history['val_loss'].append(val_loss)

        self.is_fitted = True
        return history

    def evaluate_loss(self, X: np.ndarray, y: np.ndarray, criterion: nn.Module) -> float:
        self.model.eval()
        t_X = torch.tensor(X, dtype=torch.float32).to(self.device)
        t_y = torch.tensor(y, dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits = self.model(t_X)
            loss = criterion(logits, t_y)
        return float(loss.item())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        if X.ndim == 2:
            X = np.expand_dims(X, 0)
        t_X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(t_X)
            probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            'model_type': self.model_type,
            'state_dict': self.model.state_dict(),
            'is_fitted': self.is_fitted
        }
        torch.save(payload, filepath)

    def load(self, filepath: str) -> 'DeepClassifierWrapper':
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Deep model checkpoint not found: {filepath}")
        payload = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(payload['state_dict'])
        self.is_fitted = payload['is_fitted']
        self.model.eval()
        return self
