<!-- Header Wave Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=🫀%20ECG-ID&fontSize=50&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Deep%20Learning%20ECG%20Biometric%20Identification&descSize=20&descAlignY=55" width="100%"/>

<div align="center">

<!-- Typing Animation Title -->
<a href="https://github.com/Marways7/ECG_IdentificationX">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&duration=3000&pause=1500&color=E74C3C&center=true&vCenter=true&repeat=true&width=450&height=25&lines=%E5%9F%BA%E4%BA%8E%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84ECG%E8%BA%AB%E4%BB%BD%E8%AF%86%E5%88%AB;ECG+Biometric+Identification" alt="Typing SVG" />
</a>

<br/><br/>

<!-- Badges Row 1: Social -->
<p>
<a href="https://github.com/Marways7/ECG_IdentificationX/stargazers"><img src="https://img.shields.io/github/stars/Marways7/ECG_IdentificationX?style=for-the-badge&logo=github&color=yellow&cacheSeconds=3600" alt="Stars"></a>
<a href="https://github.com/Marways7/ECG_IdentificationX/network/members"><img src="https://img.shields.io/github/forks/Marways7/ECG_IdentificationX?style=for-the-badge&logo=github&color=blue&cacheSeconds=3600" alt="Forks"></a>
<a href="https://github.com/Marways7/ECG_IdentificationX/issues"><img src="https://img.shields.io/github/issues/Marways7/ECG_IdentificationX?style=for-the-badge&logo=github&color=red&cacheSeconds=3600" alt="Issues"></a>
</p>

<!-- Badges Row 2: Tech -->
<p>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
<img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/DeepSeek-AI-00D4AA?style=flat-square&logo=openai&logoColor=white" alt="DeepSeek">
<img src="https://img.shields.io/badge/License-MIT-success?style=flat-square" alt="License">
</p>

<!-- Badges Row 3: Stats -->
<p>
<img src="https://img.shields.io/badge/Accuracy-98.44%25-brightgreen?style=flat-square&logo=checkmarx&logoColor=white" alt="Accuracy">
<img src="https://img.shields.io/badge/F1_Score-98.44%25-blue?style=flat-square" alt="F1">
<img src="https://img.shields.io/badge/Parameters-~44K-orange?style=flat-square" alt="Params">
<img src="https://img.shields.io/badge/Inference-<10ms-purple?style=flat-square" alt="Speed">
</p>

<br/>

[🚀 快速開始](#-快速開始) · [📊 性能](#-效能指標) · [🔬 算法](#-核心演算法) · [📖 文件](#-文件)

---

**🌍 語言**: [English](README.md) | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-Hant.md)

</div>

<br/>

<!-- Performance Cards -->
<div align="center">

<table>
<tr>
<td align="center" width="25%">
<img width="60" src="https://img.icons8.com/fluency/96/accuracy.png" alt="accuracy"/>
<br/><b>98.44%</b>
<br/><sub>測試準確率</sub>
</td>
<td align="center" width="25%">
<img width="60" src="https://img.icons8.com/fluency/96/combo-chart.png" alt="f1"/>
<br/><b>98.44%</b>
<br/><sub>F1分数</sub>
</td>
<td align="center" width="25%">
<img width="60" src="https://img.icons8.com/fluency/96/module.png" alt="params"/>
<br/><b>~44K</b>
<br/><sub>模型参数</sub>
</td>
<td align="center" width="25%">
<img width="60" src="https://img.icons8.com/fluency/96/speed.png" alt="speed"/>
<br/><b><10ms</b>
<br/><sub>推論時間</sub>
</td>
</tr>
</table>

</div>

---

## 📚 專案背景

### 🎓 課程專案

本專案是**生產實習課程**的實作作品，旨在探索基於心電訊號(ECG)的生物特徵身分識別技術。

### 💡 研究動機

心電圖(ECG)作為一種新興的生物特徵識別手段，相較傳統方法具有獨特優勢：

| 特性 | 優勢說明 |
|:---:|:---|
| 🔐 **活體檢測** | ECG信号只能從活體取得，天然具備防偽能力 |
| 🧬 **唯一性** | 每個人的心臟電生理特徵具有獨特性 |
| 🛡️ **難以偽造** | 相較指紋、臉部等特徵，ECG極難被複製 |
| ⏱️ **持續驗證** | 可實現持續的身分監測與驗證 |

### 📋 專案概述

- **資料採集**: 使用ADS1292R心電採集裝置收集6位同學的心電資料
- **採樣規格**: 250Hz採樣率，24位ADC精度，每人約5分鐘資料
- **核心技術**: 訊號預處理 + Pan-Tompkins R峰檢測 + 1D-CNN深度學習分類
- **最終成果**: 測試準確率達到 **98.44%**，滿足實際應用需求

---

## ✨ 專案亮點

<table>
<tr>
<td width="50%">

### 🎯 高精度識別

| 指标 | 数值 |
|:---|:---:|
| 測試準確率 | **98.44%** |
| F1分数 | **98.44%** |
| 特異性 | **99.68%** |
| 誤判數 | **8/512** |

</td>
<td width="50%">

### ⚡ 輕量高效

| 特性 | 数值 |
|:---|:---:|
| 模型参数 | **~44,000** |
| 推論時間 | **<10ms** |
| 模型體積 | **<1MB** |
| 即時驗證 | **✅** |

</td>
</tr>
<tr>
<td width="50%">

### 🔬 專業演算法

- 📈 **Pan-Tompkins** R峰检测
- 🌊 **db6小波** 自適應去噪
- ❤️ **HRV分析** 时/频/非線性
- 🎵 **MFCC** 頻域特徵提取

</td>
<td width="50%">

### 🤖 AI智慧分析

- 🧠 **DeepSeek AI** 輔助診斷
- 📝 **智慧報告** 自動生成
- 📊 **互動式** 可视化界面
- 🎨 **学术风格** 专业UI

</td>
</tr>
</table>

---

## 🖼️ 頁面展示

<div align="center">

### 📊 資料分析
<img src="screenshots/data_analysis.png" alt="資料分析" width="80%"/>
<br/><sub>訊號預處理、R峰檢測、心拍分割視覺化</sub>

<br/><br/>

### 🎯 身分識別
<img src="screenshots/identification.png" alt="身分識別" width="80%"/>
<br/><sub>实时身分識別与置信度展示</sub>

<br/><br/>

### 📈 模型評估
<img src="screenshots/model_evaluation.png" alt="模型評估" width="80%"/>
<br/><sub>混淆矩陣、準確率、F1分數等評估指標</sub>

<br/><br/>

### 🤖 AI智慧分析
<img src="screenshots/ai_analysis.png" alt="AI智慧分析" width="80%"/>
<br/><sub>DeepSeek AI輔助生成分析報告</sub>

</div>

---

## �📊 效能指標

<div align="center">

| 🎯 指标 | 📈 数值 | 🎯 指标 | 📈 数值 |
|:---:|:---:|:---:|:---:|
| **測試準確率** | 98.44% | **验证准确率** | 99.22% |
| **精确率** | 98.45% | **敏感度/召回率** | 98.44% |
| **F1分数** | 98.44% | **特異性** | 99.68% |

</div>

### 📋 混淆矩陣

<div align="center">

|  | **A** | **B** | **C** | **D** | **E** | **F** | **Recall** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** | 72 | 0 | 1 | 0 | 0 | 0 | 98.6% |
| **B** | 0 | 104 | 0 | 0 | 0 | 1 | 99.0% |
| **C** | 2 | 0 | 103 | 0 | 0 | 0 | 98.1% |
| **D** | 0 | 0 | 1 | 78 | 0 | 1 | 97.5% |
| **E** | 0 | 0 | 1 | 0 | 72 | 0 | 98.6% |
| **F** | 0 | 1 | 0 | 0 | 0 | 75 | 98.7% |

<sub>總測試樣本: 512 | 正確分類: 504 | 錯誤分類: 8</sub>

</div>

---

## 🚀 快速開始

### 📋 環境需求

<table>
<tr>
<td>

| 依赖 | 版本 |
|:---|:---:|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| CUDA | 11.8+ (可選) |

</td>
<td>

| 必要設定 | 说明 |
|:---|:---|
| DeepSeek API | AI分析功能 |
| `.env` 文件 | 存储API密钥 |

</td>
</tr>
</table>

### 📥 安裝步驟

```bash
# 1️⃣ 複製儲存庫
git clone https://github.com/Marways7/ECG_IdentificationX.git
cd ECG_IdentificationX

# 2️⃣ 安裝相依套件
pip install -r requirements.txt

# 3️⃣ 設定API金鑰
echo "DEEPSEEK_API_KEY=your_api_key_here" > .env
echo "DEEPSEEK_BASE_URL=https://api.deepseek.com/v1" >> .env

# 4️⃣ 準備ECG資料 (见下方说明)
# 將你的ECG資料檔案放入 ECG_data/ 目錄

# 5️⃣ 訓練模型
python train.py --epochs 50 --model_type lightweight

# 6️⃣ 啟動Web介面
streamlit run app.py
```

<details>
<summary>📁 <b>如何準備ECG資料？</b></summary>

> ⚠️ **隱私說明**: 由於心電資料屬於個人生物特徵資料，涉及隱私，本儲存庫不包含原始資料。

請將你的ECG資料檔案放入 `ECG_data/` 目錄，資料格式要求：

| 欄位 | 说明 |
|:---|:---|
| `timestamp` | 時間戳 |
| `Channel 1` | ECG信号 (24-bit ADC原始值) |
| `Channel 2` | 呼吸訊號 (可選) |

**範例檔案結構**:
```
ECG_data/
├── A.csv    # 受試者A的数据
├── B.csv    # 受試者B的数据
└── ...      # 更多受試者
```

**CSV文件格式**:
```csv
timestamp,Channel 1,Channel 2,Channel 3
1767759832,128363,11421861,652
1767759832,128774,11421871,652
...
```

</details>

<details>
<summary>📌 <b>如何取得 DeepSeek API Key？</b></summary>

1. 访问 [DeepSeek Platform](https://platform.deepseek.com/)
2. 註冊/登入帳號
3. 创建 API Key
4. 複製到 `.env` 文件

</details>

> 🌐 啟動後訪問 **http://localhost:8501**

---

## 🏗️ 系統架構

```mermaid
flowchart LR
    subgraph A[" 📥 採集 "]
        A1["ADS1292R<br/>250Hz"]
    end
    
    subgraph B[" 🔧 預處理 "]
        B1["滤波"] --> B2["去噪"]
    end
    
    subgraph C[" 📊 特徵 "]
        C1["R峰检测"]
        C2["HRV/MFCC"]
    end
    
    subgraph D[" 🧠 模型 "]
        D1["1D-CNN"]
    end
    
    subgraph E[" 📤 輸出 "]
        E1["身份A-F"]
    end
    
    A1 --> B1
    B2 --> C1
    C1 --> C2
    C2 --> D1
    D1 --> E1
    
    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#e8f5e9
    style D fill:#f3e5f5
    style E fill:#ffebee
```

### 📁 專案結構

```
📦 ECG_IdentificationX
 ┣ 📂 ECG_data/          # 原始資料 (A-F, 共2558心拍)
 ┣ 📂 src/
 ┃ ┣ 📂 preprocessing/   # 信号預處理
 ┃ ┣ 📂 feature_extraction/  # 特徵提取
 ┃ ┣ 📂 models/          # 深度学习模型
 ┃ ┗ 📂 utils/           # 工具函式
 ┣ 📂 docs/              # 技术文件
 ┣ 📄 app.py             # Web界面
 ┣ 📄 train.py           # 訓練腳本
 ┗ 📄 .env               # API設定
```

---

## 🔬 核心演算法

### 信号預處理

```mermaid
flowchart LR
    A["原始ADC"] --> B["DC去除"] --> C["50Hz陷波"] --> D["帶通濾波"] --> E["小波去噪"] --> F["干净ECG"]
    style A fill:#ffcdd2
    style F fill:#c8e6c9
```

### Pan-Tompkins R峰检测

```mermaid
flowchart LR
    A["ECG"] --> B["带通5-15Hz"] --> C["差分"] --> D["平方"] --> E["积分"] --> F["阈值"] --> G["R峰"]
    style A fill:#e3f2fd
    style G fill:#c8e6c9
```

### HRV分析體系

| ⏱️ 時域 | 📊 頻域 | 🔀 非線性 |
|:---:|:---:|:---:|
| SDNN | VLF | SD1/SD2 |
| RMSSD | LF | ApEn |
| pNN50 | HF | SampEn |
| pNN20 | LF/HF | DFA |

### LightweightCNN

```
输入[175] → Conv(32,k7) → Conv(64,k5) → Conv(128,k3) → GAP → FC(64) → 輸出[6]
```

---

## 📦 資料規格

<div align="center">

| 参数 | 值 | 参数 | 值 |
|:---:|:---:|:---:|:---:|
| 芯片 | ADS1292R | 采样率 | 250Hz |
| ADC | 24-bit | 增益 | 6x |
| 参考电压 | 2.42V | 總心拍數 | 2,558 |

**ADC转换公式**: $V_{mV} = \frac{(ADC - \overline{ADC}) \times 2 \times 2.42 \times 1000}{6 \times 2^{24}}$

</div>

---

## 🛠️ 技術棧

<div align="center">

<img src="https://skillicons.dev/icons?i=python,pytorch,sklearn&theme=light" alt="Tech" />

| 类别 | 技术 |
|:---:|:---|
| 🐍 語言 | Python 3.10+ |
| 🧠 深度学习 | PyTorch 2.0+ |
| 📊 訊號處理 | SciPy, NumPy, PyWavelets |
| 🤖 AI | DeepSeek API |
| 🌐 Web | Streamlit, Plotly |

</div>

---

## 📖 文件

| 📄 文件 | 📝 内容 |
|:---|:---|
| [📘 系統概述](docs/Technical_Documentation_Part1_Overview.md) | 架构、資料規格 |
| [📗 信号預處理](docs/Technical_Documentation_Part2_Preprocessing.md) | 滤波、去噪 |
| [📙 R峰检测](docs/Technical_Documentation_Part3_R_Peak_Detection.md) | Pan-Tompkins |
| [📕 特徵提取](docs/Technical_Documentation_Part4_HRV_Features.md) | HRV、MFCC |
| [📓 深度学习](docs/Technical_Documentation_Part5_Deep_Learning_Models.md) | CNN模型 |
| [📔 訓練評估](docs/Technical_Documentation_Part6_Training.md) | 训练策略 |
| [📒 性能总结](docs/Technical_Documentation_Part7_Performance_Summary.md) | 結果分析 |

---

## 🇬🇧 English

<details>
<summary><b>📖 Click to expand English documentation</b></summary>

<br/>

### ✨ Highlights

| Feature | Description |
|:---|:---|
| 🎯 **High Accuracy** | 98.44% test accuracy, 98.44% F1 score |
| ⚡ **Lightweight** | Only ~44K parameters, <10ms inference |
| 🔬 **Professional** | Pan-Tompkins + Wavelet + HRV + MFCC |
| 🤖 **AI-Powered** | DeepSeek AI intelligent analysis |

### 🚀 Quick Start

```bash
git clone https://github.com/Marways7/ECG_IdentificationX.git
cd ECG_IdentificationX
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key" > .env
python train.py
streamlit run app.py
```

### 📊 Performance

| Metric | Value | Metric | Value |
|:---:|:---:|:---:|:---:|
| Test Accuracy | 98.44% | Precision | 98.45% |
| F1 Score | 98.44% | Recall | 98.44% |

</details>

---

## 📄 License

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

This project is licensed under the **MIT License**

</div>

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Marways7/ECG_IdentificationX&type=date&legend=top-left)](https://www.star-history.com/#Marways7/ECG_IdentificationX&type=date&legend=top-left)

---

<div align="center">

### ⭐ 如果這個專案對你有幫助，請點個 Star！

<a href="https://github.com/Marways7/ECG_IdentificationX/stargazers">
<img src="https://img.shields.io/github/stars/Marways7/ECG_IdentificationX?style=for-the-badge&logo=github&color=yellow&cacheSeconds=3600" alt="Stars">
</a>

<br/><br/>

**Made with ❤️ for ECG Biometric Research**

<sub>© 2026 ECG-ID Project | <a href="https://github.com/Marways7">@Marways7</a></sub>

</div>

<!-- Footer Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>
