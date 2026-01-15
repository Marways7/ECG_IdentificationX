# 第七部分：系统性能与总结

## 7.1 最终性能指标

### 7.1.1 整体性能

| 指标 | 目标值 | 实际值 | 状态 |
|------|--------|--------|------|
| **测试准确率** | ≥95% | **98.44%** | ✅ 超越目标 |
| **验证准确率** | - | **99.22%** | ✅ |
| **精确率** | ≥93% | **98.45%** | ✅ 超越目标 |
| **敏感度(召回率)** | ≥93% | **98.44%** | ✅ 超越目标 |
| **F1分数** | ≥94% | **98.44%** | ✅ 超越目标 |

### 7.1.2 混淆矩阵详细分析

```
          预测类别
真实     A     B     C     D     E     F
类别  ┌─────┬─────┬─────┬─────┬─────┬─────┐
  A   │ 72  │  0  │  1  │  0  │  0  │  0  │  正确率: 98.6%
      ├─────┼─────┼─────┼─────┼─────┼─────┤
  B   │  0  │ 104 │  0  │  0  │  0  │  1  │  正确率: 99.0%
      ├─────┼─────┼─────┼─────┼─────┼─────┤
  C   │  2  │  0  │ 103 │  0  │  0  │  0  │  正确率: 98.1%
      ├─────┼─────┼─────┼─────┼─────┼─────┤
  D   │  0  │  0  │  1  │ 78  │  0  │  1  │  正确率: 97.5%
      ├─────┼─────┼─────┼─────┼─────┼─────┤
  E   │  0  │  0  │  1  │  0  │ 72  │  0  │  正确率: 98.6%
      ├─────┼─────┼─────┼─────┼─────┼─────┤
  F   │  0  │  1  │  0  │  0  │  0  │ 75  │  正确率: 98.7%
      └─────┴─────┴─────┴─────┴─────┴─────┘
```

### 7.1.3 各受试者性能分析

| 受试者 | 样本数 | 正确数 | 准确率 | 精确率 | 敏感度 | 特异性 |
|--------|--------|--------|--------|--------|--------|--------|
| A | 73 | 72 | 98.6% | 97.3% | 98.6% | 99.5% |
| B | 105 | 104 | 99.0% | 99.0% | 99.0% | 99.8% |
| C | 105 | 103 | 98.1% | 97.2% | 98.1% | 99.3% |
| D | 80 | 78 | 97.5% | 100% | 97.5% | 100% |
| E | 73 | 72 | 98.6% | 100% | 98.6% | 100% |
| F | 76 | 75 | 98.7% | 97.4% | 98.7% | 99.5% |
| **平均** | - | - | **98.44%** | **98.45%** | **98.44%** | **99.68%** |

### 7.1.4 误分类分析

总误分类数：**8例** / 512测试样本 = **1.56%错误率**

| 误分类情况 | 次数 | 可能原因 |
|------------|------|----------|
| A→C | 1 | 形态相似性 |
| B→F | 1 | 心率区间重叠 |
| C→A | 2 | 形态相似性 |
| D→C | 1 | 形态相似性 |
| D→F | 1 | 心率区间重叠 |
| E→C | 1 | 形态相似性 |
| F→B | 1 | 心率区间重叠 |

---

## 7.2 数据集统计

### 7.2.1 各受试者心拍分布

| 受试者 | 心拍总数 | 训练集 | 验证集 | 测试集 | 占比 |
|--------|----------|--------|--------|--------|------|
| A | 366 | 256 | 37 | 73 | 14.3% |
| B | 525 | 368 | 52 | 105 | 20.5% |
| C | 522 | 366 | 51 | 105 | 20.4% |
| D | 400 | 280 | 40 | 80 | 15.6% |
| E | 363 | 254 | 36 | 73 | 14.2% |
| F | 382 | 266 | 40 | 76 | 14.9% |
| **总计** | **2,558** | **1,790** | **256** | **512** | **100%** |

### 7.2.2 心率分布统计

| 受试者 | 平均心率 | 心率范围 | 心率变异性 |
|--------|----------|----------|------------|
| A | 102.1 BPM | 95-110 | 中等 |
| B | 119.2 BPM | 110-128 | 较高 |
| C | 114.1 BPM | 105-122 | 中等 |
| D | 86.4 BPM | 80-93 | 较低 |
| E | 74.9 BPM | 70-82 | 较低 |
| F | 89.5 BPM | 82-97 | 中等 |

---

## 7.3 算法性能对比

### 7.3.1 不同模型架构对比

| 模型 | 参数量 | 测试准确率 | 推理时间 | 推荐场景 |
|------|--------|------------|----------|----------|
| LightweightCNN | ~44K | **98.44%** | <10ms | 实时识别 ⭐ |
| ECG1DCNN | ~150K | 97.85% | ~15ms | 高精度需求 |
| TDNN | ~200K | 97.26% | ~20ms | 时序分析 |
| MFCCClassifier | ~50K | 95.12% | <5ms | 频域分类 |
| FusionClassifier | ~500K | 98.05% | ~50ms | 多模态融合 |

**结论**: LightweightCNN在最少参数量下达到最高准确率，是最佳选择。

### 7.3.2 消融实验

| 实验配置 | 准确率 | Δ |
|----------|--------|---|
| 完整流程 (baseline) | 98.44% | - |
| 无小波去噪 | 95.31% | -3.13% |
| 无异常值去除 | 96.48% | -1.96% |
| 无Z-score标准化 | 94.14% | -4.30% |
| 无数据增强 | 97.66% | -0.78% |
| 无Dropout | 96.87% | -1.57% |

---

## 7.4 系统特点总结

### 7.4.1 技术亮点

1. **完整的信号处理流水线**
   - ADC转换 → 异常值去除 → 陷波滤波 → 带通滤波 → 小波去噪
   - SNR提升 >15dB

2. **改进的Pan-Tompkins R峰检测**
   - 结合后处理的异常R峰去除和遗漏R峰填补
   - 检测准确率 >98%

3. **全面的HRV分析**
   - 时域: SDNN, RMSSD, pNN50, pNN20, SDSD, CV
   - 频域: VLF, LF, HF, LF/HF
   - 非线性: SD1, SD2, ApEn, SampEn, DFA

4. **高效的深度学习模型**
   - 轻量级1D-CNN，仅~44K参数
   - 测试准确率98.44%
   - 推理时间<10ms

5. **专业级可视化界面**
   - 学术期刊风格设计
   - 交互式Plotly图表
   - 实时信号分析展示

### 7.4.2 创新点

| 创新点 | 描述 |
|--------|------|
| 自适应小波阈值 | 层级相关的自适应阈值计算 |
| 改进R峰检测 | 后处理填补和去除策略 |
| 轻量级CNN | 高准确率与低计算量的最佳平衡 |
| 学术风格UI | 专业、美观的可视化设计 |

---

## 7.5 使用指南

### 7.5.1 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 主要依赖
# - Python 3.10+
# - PyTorch 2.0+
# - SciPy, NumPy, PyWavelets
# - Streamlit, Plotly
# - scikit-learn, pandas
```

### 7.5.2 训练模型

```bash
# 默认训练 (轻量级CNN)
python train.py

# 自定义参数
python train.py --epochs 50 --model_type lightweight --batch_size 32

# 可选模型类型
# - lightweight (默认，推荐)
# - cnn
# - tdnn
# - mfcc
# - fusion
```

### 7.5.3 启动Web界面

```bash
streamlit run app.py
```

访问 http://localhost:8501

### 7.5.4 API使用示例

```python
from src.utils.data_loader import ECGDataLoader
from src.models.ecg_classifier import create_model
from src.models.trainer import ECGTrainer

# 加载数据
loader = ECGDataLoader('ECG_data')
beats, mfcc_features, labels, class_names = loader.prepare_dataset()

# 创建模型
model = create_model('lightweight', beat_length=175, num_classes=6)

# 训练
trainer = ECGTrainer(model)
trainer.train(train_loader, val_loader, epochs=50)

# 评估
results = trainer.evaluate(test_loader, class_names)
print(f"准确率: {results['accuracy']:.4f}")
```

---

## 7.6 未来改进方向

### 7.6.1 算法层面

1. **数据增强**
   - 时间拉伸/压缩
   - 添加模拟噪声
   - 信号翻转

2. **模型改进**
   - Transformer编码器
   - 自监督预训练
   - 域自适应

3. **特征工程**
   - 更多HRV指标
   - 形态学特征细化
   - 特征选择优化

### 7.6.2 系统层面

1. **实时采集**
   - 蓝牙实时数据流
   - 在线识别

2. **移动端部署**
   - 模型量化
   - ONNX导出
   - 移动端推理

3. **安全性**
   - 活体检测
   - 防欺骗机制

---

## 7.7 参考文献

1. Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, (3), 230-236.

2. Odinaka, I., et al. (2012). ECG biometric recognition: A comparative analysis. *IEEE Transactions on Information Forensics and Security*, 7(6), 1812-1824.

3. Malik, M., et al. (1996). Heart rate variability: Standards of measurement, physiological interpretation, and clinical use. *European Heart Journal*, 17(3), 354-381.

4. Kamath, M. V., & Fallen, E. L. (1993). Power spectral analysis of heart rate variability: a noninvasive signature of cardiac autonomic function. *Critical Reviews in Biomedical Engineering*, 21(3), 245-311.

5. Donoho, D. L., & Johnstone, I. M. (1994). Ideal spatial adaptation by wavelet shrinkage. *Biometrika*, 81(3), 425-455.

---

## 7.8 致谢

本项目作为生产实习课程项目，感谢：
- 指导教师的悉心指导
- A-F六位同学提供的心电数据
- ADS1292R硬件平台支持

---

**文档结束**

*ECG身份识别系统技术文档 v1.0*  
*编制日期: 2026年1月14日*
