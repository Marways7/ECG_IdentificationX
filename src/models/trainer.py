"""
模型训练模块
实现完整的训练流程、评估和保存
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_recall_fscore_support,
                             classification_report)
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime


class ECGDataset(Dataset):
    """
    ECG心拍数据集
    """

    def __init__(self, beats: np.ndarray, mfcc_features: np.ndarray,
                 labels: np.ndarray):
        """
        初始化数据集

        Args:
            beats: 心拍信号 [N, beat_length]
            mfcc_features: MFCC特征 [N, mfcc_dim]
            labels: 标签 [N]
        """
        self.beats = torch.FloatTensor(beats)
        self.mfcc = torch.FloatTensor(mfcc_features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'beat': self.beats[idx],
            'mfcc': self.mfcc[idx],
            'label': self.labels[idx]
        }


class ECGTrainer:
    """
    ECG身份识别模型训练器
    """

    def __init__(self, model: nn.Module,
                 device: str = 'auto',
                 save_dir: str = 'models/saved'):
        """
        初始化训练器

        Args:
            model: PyTorch模型
            device: 设备 ('cpu', 'cuda', 'auto')
            save_dir: 模型保存目录
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train(self, train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              lr: float = 0.001,
              weight_decay: float = 1e-4,
              patience: int = 15) -> Dict:
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值

        Returns:
            训练历史
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr,
                                weight_decay=weight_decay)

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_acc = 0
        no_improve = 0

        for epoch in range(epochs):
            # 训练阶段
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer
            )

            # 验证阶段
            val_loss, val_acc = self._validate(val_loader, criterion)

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 学习率调度
            scheduler.step(val_loss)

            print(f'Epoch {epoch+1}/{epochs} - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_model.pth')
                no_improve = 0
            else:
                no_improve += 1

            # 早停
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # 加载最佳模型
        self.load_model('best_model.pth')

        return self.history

    def _train_epoch(self, train_loader: DataLoader,
                     criterion: nn.Module,
                     optimizer: optim.Optimizer) -> Tuple[float, float]:
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            beats = batch['beat'].to(self.device)
            mfcc = batch['mfcc'].to(self.device)
            labels = batch['label'].to(self.device)

            optimizer.zero_grad()

            # 根据模型类型选择输入
            if hasattr(self.model, 'fusion'):
                outputs = self.model(beats, mfcc)
            elif 'mfcc' in str(type(self.model)).lower():
                outputs = self.model(mfcc)
            else:
                outputs = self.model(beats)

            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return total_loss / len(train_loader), correct / total

    def _validate(self, val_loader: DataLoader,
                  criterion: nn.Module) -> Tuple[float, float]:
        """
        验证模型
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                beats = batch['beat'].to(self.device)
                mfcc = batch['mfcc'].to(self.device)
                labels = batch['label'].to(self.device)

                if hasattr(self.model, 'fusion'):
                    outputs = self.model(beats, mfcc)
                elif 'mfcc' in str(type(self.model)).lower():
                    outputs = self.model(mfcc)
                else:
                    outputs = self.model(beats)

                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return total_loss / len(val_loader), correct / total

    def evaluate(self, test_loader: DataLoader,
                 class_names: Optional[List[str]] = None) -> Dict:
        """
        评估模型

        Args:
            test_loader: 测试数据加载器
            class_names: 类别名称

        Returns:
            评估结果字典
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                beats = batch['beat'].to(self.device)
                mfcc = batch['mfcc'].to(self.device)
                labels = batch['label'].to(self.device)

                if hasattr(self.model, 'fusion'):
                    outputs = self.model(beats, mfcc)
                elif 'mfcc' in str(type(self.model)).lower():
                    outputs = self.model(mfcc)
                else:
                    outputs = self.model(beats)

                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 计算指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        conf_matrix = confusion_matrix(all_labels, all_preds)

        # 计算每个类别的指标
        per_class_metrics = {}
        precision_per, recall_per, f1_per, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None
        )

        if class_names is None:
            class_names = [f'Class_{i}' for i in range(len(precision_per))]

        for i, name in enumerate(class_names):
            # 特异性计算
            tn = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - \
                 np.sum(conf_matrix[:, i]) + conf_matrix[i, i]
            fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            per_class_metrics[name] = {
                'precision': precision_per[i],
                'recall': recall_per[i],  # 敏感度
                'f1': f1_per[i],
                'specificity': specificity,
                'support': int(support[i])
            }

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,  # 平均敏感度
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'per_class_metrics': per_class_metrics,
            'predictions': all_preds.tolist(),
            'true_labels': all_labels.tolist()
        }

        # 打印报告
        print("\n" + "="*50)
        print("模型评估报告")
        print("="*50)
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"精确率 (Precision): {precision:.4f}")
        print(f"敏感度 (Recall/Sensitivity): {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print("\n混淆矩阵:")
        print(conf_matrix)
        print("\n各类别详细指标:")
        for name, metrics in per_class_metrics.items():
            print(f"  {name}: "
                  f"精确率={metrics['precision']:.3f}, "
                  f"敏感度={metrics['recall']:.3f}, "
                  f"特异性={metrics['specificity']:.3f}")

        return results

    def save_model(self, filename: str):
        """保存模型"""
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)

    def load_model(self, filename: str):
        """加载模型"""
        path = os.path.join(self.save_dir, filename)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'history' in checkpoint:
                self.history = checkpoint['history']

    def predict(self, beat: np.ndarray,
                mfcc: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        单样本预测

        Args:
            beat: 心拍信号 [length]
            mfcc: MFCC特征 [mfcc_dim]

        Returns:
            (预测类别, 概率分布)
        """
        self.model.eval()

        beat_tensor = torch.FloatTensor(beat).unsqueeze(0).to(self.device)
        mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if hasattr(self.model, 'fusion'):
                outputs = self.model(beat_tensor, mfcc_tensor)
            elif 'mfcc' in str(type(self.model)).lower():
                outputs = self.model(mfcc_tensor)
            else:
                outputs = self.model(beat_tensor)

            probs = torch.softmax(outputs, dim=1)
            pred = outputs.argmax(dim=1).item()

        return pred, probs.cpu().numpy()[0]


def prepare_data(beats: np.ndarray,
                 mfcc_features: np.ndarray,
                 labels: np.ndarray,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 batch_size: int = 32,
                 random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    准备数据加载器

    Args:
        beats: 心拍信号
        mfcc_features: MFCC特征
        labels: 标签
        test_size: 测试集比例
        val_size: 验证集比例
        batch_size: 批大小
        random_state: 随机种子

    Returns:
        (训练加载器, 验证加载器, 测试加载器)
    """
    # 分层划分
    X_temp_beats, X_test_beats, X_temp_mfcc, X_test_mfcc, y_temp, y_test = \
        train_test_split(beats, mfcc_features, labels,
                         test_size=test_size,
                         stratify=labels,
                         random_state=random_state)

    val_ratio = val_size / (1 - test_size)
    X_train_beats, X_val_beats, X_train_mfcc, X_val_mfcc, y_train, y_val = \
        train_test_split(X_temp_beats, X_temp_mfcc, y_temp,
                         test_size=val_ratio,
                         stratify=y_temp,
                         random_state=random_state)

    # 创建数据集
    train_dataset = ECGDataset(X_train_beats, X_train_mfcc, y_train)
    val_dataset = ECGDataset(X_val_beats, X_val_mfcc, y_val)
    test_dataset = ECGDataset(X_test_beats, X_test_mfcc, y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
