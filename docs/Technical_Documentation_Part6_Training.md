# 第六部分：模型训练与评估

## 6.1 训练流程概述

### 6.1.1 训练流水线

```
数据准备
    │
    ├── 加载CSV数据
    ├── 信号预处理
    ├── R峰检测 + 心拍分割
    ├── MFCC特征提取
    └── 分层数据集划分 (7:1:2)
         │
         ▼
模型训练
    │
    ├── 创建DataLoader
    ├── 初始化模型和优化器
    ├── 训练循环 (epochs)
    │   ├── 前向传播
    │   ├── 损失计算
    │   ├── 反向传播
    │   └── 参数更新
    ├── 验证循环
    ├── 学习率调度
    └── 早停检查
         │
         ▼
模型评估
    │
    ├── 测试集预测
    ├── 混淆矩阵
    ├── 准确率/精确率/召回率/F1
    └── 保存模型和报告
```

---

## 6.2 数据集类实现

### 6.2.1 ECGDataset

```python
from torch.utils.data import Dataset, DataLoader

class ECGDataset(Dataset):
    """ECG心拍数据集"""
    
    def __init__(self, beats, mfcc_features, labels):
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
```

### 6.2.2 数据划分函数

```python
from sklearn.model_selection import train_test_split

def prepare_data(beats, mfcc_features, labels,
                 test_size=0.2, val_size=0.1,
                 batch_size=32, random_state=42):
    """
    准备数据加载器
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 第一次划分：分离测试集
    X_temp_beats, X_test_beats, X_temp_mfcc, X_test_mfcc, y_temp, y_test = \
        train_test_split(beats, mfcc_features, labels,
                         test_size=test_size,
                         stratify=labels,      # 分层抽样
                         random_state=random_state)
    
    # 第二次划分：分离验证集
    val_ratio = val_size / (1 - test_size)  # 0.1 / 0.8 = 0.125
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
```

---

## 6.3 损失函数

### 6.3.1 交叉熵损失

多分类任务使用**交叉熵损失 (Cross-Entropy Loss)**：

$$
\mathcal{L} = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})
$$

其中：
- $N$: 样本数
- $C$: 类别数 (6)
- $y_{ic}$: 真实标签 (one-hot)
- $\hat{y}_{ic}$: 预测概率 (softmax输出)

```python
criterion = nn.CrossEntropyLoss()
```

---

## 6.4 优化器

### 6.4.1 AdamW优化器

使用**AdamW**优化器（Adam with decoupled weight decay）：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t = \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)
$$

**超参数设置**：
```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,           # 学习率
    betas=(0.9, 0.999), # 动量参数
    eps=1e-8,           # 数值稳定性
    weight_decay=1e-4   # 权重衰减 (L2正则化)
)
```

---

## 6.5 学习率调度

### 6.5.1 ReduceLROnPlateau

当验证损失不再下降时，自动降低学习率：

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # 监控指标：最小化
    factor=0.5,      # 学习率衰减因子
    patience=5,      # 等待epoch数
    verbose=True
)

# 每个epoch后更新
scheduler.step(val_loss)
```

---

## 6.6 训练循环

### 6.6.1 单Epoch训练

```python
def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        beats = batch['beat'].to(device)
        mfcc = batch['mfcc'].to(device)
        labels = batch['label'].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播（根据模型类型选择输入）
        if hasattr(model, 'fusion'):
            outputs = model(beats, mfcc)
        elif 'mfcc' in str(type(model)).lower():
            outputs = model(mfcc)
        else:
            outputs = model(beats)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 参数更新
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), correct / total
```

### 6.6.2 验证循环

```python
def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            beats = batch['beat'].to(device)
            mfcc = batch['mfcc'].to(device)
            labels = batch['label'].to(device)
            
            if hasattr(model, 'fusion'):
                outputs = model(beats, mfcc)
            elif 'mfcc' in str(type(model)).lower():
                outputs = model(mfcc)
            else:
                outputs = model(beats)
            
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(val_loader), correct / total
```

---

## 6.7 早停策略

### 6.7.1 早停实现

防止过拟合，当验证指标不再提升时停止训练：

```python
def train_with_early_stopping(model, train_loader, val_loader,
                               epochs=100, patience=15):
    best_val_acc = 0
    no_improve = 0
    
    for epoch in range(epochs):
        # 训练
        train_loss, train_acc = train_epoch(...)
        
        # 验证
        val_loss, val_acc = validate(...)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model('best_model.pth')
            no_improve = 0
        else:
            no_improve += 1
        
        # 早停检查
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # 加载最佳模型
    load_model('best_model.pth')
```

---

## 6.8 评估指标

### 6.8.1 混淆矩阵

$$
\text{Confusion Matrix} = \begin{bmatrix}
TP_A & FP_{AB} & \cdots \\
FP_{BA} & TP_B & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}
$$

### 6.8.2 评估指标定义

| 指标 | 公式 | 意义 |
|------|------|------|
| **准确率 (Accuracy)** | $\frac{TP+TN}{TP+TN+FP+FN}$ | 总体正确率 |
| **精确率 (Precision)** | $\frac{TP}{TP+FP}$ | 预测为正的正确率 |
| **召回率/敏感度 (Recall)** | $\frac{TP}{TP+FN}$ | 实际正例的检出率 |
| **特异性 (Specificity)** | $\frac{TN}{TN+FP}$ | 实际负例的正确拒绝率 |
| **F1分数** | $\frac{2 \times P \times R}{P + R}$ | 精确率和召回率的调和平均 |

### 6.8.3 评估函数实现

```python
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_recall_fscore_support)

def evaluate(model, test_loader, class_names, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            beats = batch['beat'].to(device)
            mfcc = batch['mfcc'].to(device)
            labels = batch['label'].to(device)
            
            if hasattr(model, 'fusion'):
                outputs = model(beats, mfcc)
            else:
                outputs = model(beats)
            
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
    
    # 计算每个类别的特异性
    per_class_metrics = {}
    for i, name in enumerate(class_names):
        tn = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - \
             np.sum(conf_matrix[:, i]) + conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        per_class_metrics[name] = {
            'precision': precision_per[i],
            'recall': recall_per[i],
            'specificity': specificity
        }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'per_class_metrics': per_class_metrics
    }
```

---

## 6.9 训练配置

### 6.9.1 默认超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 100 | 最大训练轮数 |
| batch_size | 32 | 批大小 |
| learning_rate | 0.001 | 初始学习率 |
| weight_decay | 1e-4 | 权重衰减 |
| patience | 15 | 早停耐心值 |
| dropout | 0.3 | Dropout概率 |

### 6.9.2 命令行训练

```bash
python train.py --epochs 50 --model_type lightweight --batch_size 32
```

---

*[继续第七部分: 系统性能与总结]*
