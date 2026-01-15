# ECG身份识别系统完整技术文档

## Complete Technical Documentation for ECG-Based Biometric Identification System

---

**文档版本**: v1.0  
**编制日期**: 2026年1月14日  
**系统版本**: ECG_IdentificationX v1.0  
**项目性质**: 生产实习课程项目

---

## 📑 文档目录

本技术文档分为七个部分，全面阐述ECG身份识别系统的算法原理、实现细节和性能评估。

### [第一部分：系统概述与技术架构](Technical_Documentation_Part1_Overview.md)

- 项目背景与目标
- 数据采集规格 (ADS1292R)
- ADC值转换公式
- 系统架构图
- 模块结构
- 技术栈选型
- 数据集统计与划分

### [第二部分：信号预处理算法原理](Technical_Documentation_Part2_Preprocessing.md)

- 预处理流程概述
- 异常值去除 (MAD方法)
- 陷波滤波 (50Hz)
- 带通滤波 (Butterworth)
- 小波去噪 (db6, VisuShrink)
- 信噪比评估

### [第三部分：R峰检测与心拍分割](Technical_Documentation_Part3_R_Peak_Detection.md)

- Pan-Tompkins算法详解
- 五点差分求导
- 信号平方与滑动窗口积分
- 自适应阈值检测
- 改进的R峰检测器
- 心拍分割策略
- 心率与呼吸速率计算

### [第四部分：HRV分析与特征提取](Technical_Documentation_Part4_HRV_Features.md)

- 心率变异性概述
- 时域HRV指标 (SDNN, RMSSD, pNN50等)
- 频域HRV指标 (VLF, LF, HF, LF/HF)
- 非线性HRV指标 (Poincaré, ApEn, SampEn, DFA)
- MFCC特征提取原理
- 特征融合策略

### [第五部分：深度学习模型架构](Technical_Documentation_Part5_Deep_Learning_Models.md)

- LightweightCNN (轻量级CNN)
- ECG1DCNN (残差网络)
- TDNN (时延神经网络)
- MFCCClassifier (BP神经网络)
- FusionClassifier (多模态融合)
- 模型参数对比
- 正则化策略

### [第六部分：模型训练与评估](Technical_Documentation_Part6_Training.md)

- 训练流程概述
- 数据集类实现
- 损失函数 (交叉熵)
- 优化器 (AdamW)
- 学习率调度
- 训练循环实现
- 早停策略
- 评估指标定义与实现

### [第七部分：系统性能与总结](Technical_Documentation_Part7_Performance_Summary.md)

- 最终性能指标
- 混淆矩阵详细分析
- 各受试者性能分析
- 算法性能对比
- 消融实验
- 系统特点总结
- 使用指南
- 未来改进方向

---

## 📊 核心性能指标

| 指标 | 值 |
|------|-----|
| **测试准确率** | 98.44% |
| **验证准确率** | 99.22% |
| **精确率** | 98.45% |
| **敏感度** | 98.44% |
| **F1分数** | 98.44% |
| **模型参数** | ~44K |
| **推理时间** | <10ms |

---

## 🔧 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python train.py --epochs 50 --model_type lightweight
```

### 启动Web界面

```bash
streamlit run app.py
```

---

## 📂 项目结构

```
ECG_IdentificationX/
├── ECG_data/              # 原始ECG数据 (A-F 6位受试者)
├── src/
│   ├── preprocessing/     # 信号预处理模块
│   ├── feature_extraction/# 特征提取模块
│   ├── models/           # 深度学习模型
│   └── utils/            # 工具函数
├── docs/                 # 技术文档 (本目录)
├── models/saved/         # 保存的模型权重
├── outputs/              # 输出结果
├── app.py               # Streamlit Web界面
├── train.py             # 训练脚本
└── requirements.txt     # 依赖包
```

---

## 📖 算法流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ECG身份识别系统流程                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  原始ECG → ADC转换 → 异常值去除 → 陷波滤波 → 带通滤波 → 小波去噪          │
│                                                                ↓        │
│                                                           干净ECG       │
│                                                                ↓        │
│                                                         Pan-Tompkins    │
│                                                           R峰检测        │
│                                                                ↓        │
│                                                          心拍分割       │
│                                                                ↓        │
│                                        ┌───────────────────────┼───────┐│
│                                        ↓                       ↓       ││
│                                   原始心拍信号            MFCC特征提取   ││
│                                        ↓                       ↓       ││
│                                   1D-CNN网络            BP神经网络     ││
│                                        ↓                       ↓       ││
│                                        └───────────┬───────────┘       ││
│                                                    ↓                   ││
│                                              特征融合                   ││
│                                                    ↓                   ││
│                                              身份分类                   ││
│                                                    ↓                   ││
│                                           识别结果 (A-F)               ││
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📚 参考文献

1. Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm.
2. Odinaka, I., et al. (2012). ECG biometric recognition: A comparative analysis.
3. Malik, M., et al. (1996). Heart rate variability: Standards of measurement.
4. Donoho, D. L., & Johnstone, I. M. (1994). Ideal spatial adaptation by wavelet shrinkage.

---

**ECG身份识别系统技术文档**  
*生产实习课程项目*  
*2026年1月*
