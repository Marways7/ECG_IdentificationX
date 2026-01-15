"""
ECG身份识别深度学习模型
实现1D-CNN、TDNN和融合网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Conv1DBlock(nn.Module):
    """
    1D卷积块：Conv -> BatchNorm -> ReLU -> Dropout
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 5, stride: int = 1,
                 padding: int = 2, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class ResidualBlock1D(nn.Module):
    """
    1D残差块
    """

    def __init__(self, channels: int, kernel_size: int = 5,
                 dropout: float = 0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = x + residual
        x = F.relu(x)
        return x


class ECG1DCNN(nn.Module):
    """
    1D-CNN时域特征提取网络
    用于从原始心拍信号中提取深度特征
    """

    def __init__(self, input_length: int = 175,
                 num_classes: int = 6,
                 feature_dim: int = 128):
        super().__init__()

        self.feature_dim = feature_dim

        # 第一层卷积
        self.conv1 = Conv1DBlock(1, 32, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(2)

        # 第二层卷积
        self.conv2 = Conv1DBlock(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)

        # 残差块
        self.res1 = ResidualBlock1D(64)
        self.res2 = ResidualBlock1D(64)

        # 第三层卷积
        self.conv3 = Conv1DBlock(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)

        # 第四层卷积
        self.conv4 = Conv1DBlock(128, 256, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc1 = nn.Linear(256, feature_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入 [batch, length] 或 [batch, 1, length]

        Returns:
            分类logits [batch, num_classes]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, length]

        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.res1(x)
        x = self.res2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.global_pool(x)

        x = x.squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取深度特征 (不包含最后的分类层)

        Args:
            x: 输入

        Returns:
            特征向量 [batch, feature_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        features = F.relu(self.fc1(x))

        return features


class TDNN(nn.Module):
    """
    Time-Delay Neural Network (TDNN)
    用于捕获时序依赖关系的特征
    """

    def __init__(self, input_length: int = 175,
                 num_classes: int = 6,
                 feature_dim: int = 128):
        super().__init__()

        self.feature_dim = feature_dim

        # TDNN层 - 不同的时间上下文窗口
        self.tdnn1 = nn.Conv1d(1, 64, kernel_size=5, dilation=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)

        self.tdnn2 = nn.Conv1d(64, 128, kernel_size=3, dilation=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        self.tdnn3 = nn.Conv1d(128, 256, kernel_size=3, dilation=3, padding=3)
        self.bn3 = nn.BatchNorm1d(256)

        self.tdnn4 = nn.Conv1d(256, 256, kernel_size=3, dilation=4, padding=4)
        self.bn4 = nn.BatchNorm1d(256)

        # 统计池化
        self.stats_pool = StatsPooling()

        # 全连接层
        self.fc1 = nn.Linear(512, feature_dim)
        self.bn5 = nn.BatchNorm1d(feature_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.tdnn1(x)))
        x = F.relu(self.bn2(self.tdnn2(x)))
        x = F.relu(self.bn3(self.tdnn3(x)))
        x = F.relu(self.bn4(self.tdnn4(x)))

        x = self.stats_pool(x)

        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.tdnn1(x)))
        x = F.relu(self.bn2(self.tdnn2(x)))
        x = F.relu(self.bn3(self.tdnn3(x)))
        x = F.relu(self.bn4(self.tdnn4(x)))

        x = self.stats_pool(x)
        features = F.relu(self.bn5(self.fc1(x)))

        return features


class StatsPooling(nn.Module):
    """
    统计池化层 - 计算均值和标准差
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        return torch.cat([mean, std], dim=1)


class MFCCClassifier(nn.Module):
    """
    MFCC特征分类网络 (BP神经网络)
    用于处理频域MFCC特征
    """

    def __init__(self, input_dim: int = 52,
                 num_classes: int = 6,
                 hidden_dims: Tuple[int, ...] = (256, 128, 64)):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        self.feature_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


class FusionClassifier(nn.Module):
    """
    多模态融合分类器
    融合时域CNN特征、TDNN特征和频域MFCC特征
    """

    def __init__(self, beat_length: int = 175,
                 mfcc_dim: int = 52,
                 num_classes: int = 6):
        super().__init__()

        self.cnn_feature_dim = 128
        self.tdnn_feature_dim = 128
        self.mfcc_feature_dim = 64

        # 子网络
        self.cnn = ECG1DCNN(beat_length, num_classes, self.cnn_feature_dim)
        self.tdnn = TDNN(beat_length, num_classes, self.tdnn_feature_dim)
        self.mfcc_net = MFCCClassifier(mfcc_dim, num_classes, (128, 64))

        # 融合层
        total_features = self.cnn_feature_dim + self.tdnn_feature_dim + self.mfcc_feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(total_features, total_features // 4),
            nn.ReLU(),
            nn.Linear(total_features // 4, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, beat: torch.Tensor,
                mfcc: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            beat: 心拍信号 [batch, length]
            mfcc: MFCC特征 [batch, mfcc_dim]

        Returns:
            分类logits [batch, num_classes]
        """
        # 提取各分支特征
        cnn_feat = self.cnn.extract_features(beat)
        tdnn_feat = self.tdnn.extract_features(beat)
        mfcc_feat = self.mfcc_net.extract_features(mfcc)

        # 拼接特征
        combined = torch.cat([cnn_feat, tdnn_feat, mfcc_feat], dim=1)

        # 注意力加权
        attn_weights = self.attention(combined)
        weighted_cnn = cnn_feat * attn_weights[:, 0:1]
        weighted_tdnn = tdnn_feat * attn_weights[:, 1:2]
        weighted_mfcc = mfcc_feat * attn_weights[:, 2:3]

        # 融合
        fused = torch.cat([weighted_cnn, weighted_tdnn, weighted_mfcc], dim=1)

        # 分类
        logits = self.fusion(fused)

        return logits


class LightweightCNN(nn.Module):
    """
    轻量级CNN - 用于快速推理
    """

    def __init__(self, input_length: int = 175,
                 num_classes: int = 6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(model_type: str = 'fusion',
                 beat_length: int = 175,
                 mfcc_dim: int = 52,
                 num_classes: int = 6) -> nn.Module:
    """
    创建模型

    Args:
        model_type: 模型类型 ('cnn', 'tdnn', 'mfcc', 'fusion', 'lightweight')
        beat_length: 心拍长度
        mfcc_dim: MFCC特征维度
        num_classes: 类别数

    Returns:
        模型实例
    """
    if model_type == 'cnn':
        return ECG1DCNN(beat_length, num_classes)
    elif model_type == 'tdnn':
        return TDNN(beat_length, num_classes)
    elif model_type == 'mfcc':
        return MFCCClassifier(mfcc_dim, num_classes)
    elif model_type == 'fusion':
        return FusionClassifier(beat_length, mfcc_dim, num_classes)
    elif model_type == 'lightweight':
        return LightweightCNN(beat_length, num_classes)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
