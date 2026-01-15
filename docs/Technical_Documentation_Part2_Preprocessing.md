# 第二部分：信号预处理算法原理

## 2.1 预处理流程概述

ECG信号预处理是身份识别的基础，其目标是在去除噪声的同时保留个体特征信息。

### 2.1.1 完整预处理流水线

```
原始ADC数据
     │
     ▼
┌─────────────────┐
│  ADC → 电压转换  │  去除DC偏移，转换为物理量(mV)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   边缘数据去除   │  舍弃首尾各1秒数据（采集不稳定）
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   异常值去除     │  MAD方法检测和限制极端值
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   50Hz陷波滤波   │  去除工频干扰
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   带通滤波       │  0.5-40Hz Butterworth滤波器
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   小波去噪       │  db6小波，自适应阈值
└────────┬────────┘
         │
         ▼
    干净ECG信号
```

---

## 2.2 异常值去除 (MAD方法)

### 2.2.1 问题背景

由于蓝牙传输和ADC采集特性，原始信号中可能存在**采集spike**（瞬时异常大值），这些异常值会严重影响后续处理。

### 2.2.2 MAD方法原理

**MAD (Median Absolute Deviation)** 是一种稳健的离散程度估计方法，相比标准差，它对异常值不敏感。

**数学定义**：
$$
MAD = \text{median}(|X_i - \text{median}(X)|)
$$

**标准差估计**（假设正态分布）：
$$
\hat{\sigma} = 1.4826 \times MAD
$$

其中1.4826是使MAD成为正态分布标准差无偏估计的比例因子。

### 2.2.3 算法实现

```python
def remove_outliers(data, threshold=5.0):
    """
    使用MAD方法去除异常值
    
    Args:
        data: 输入信号
        threshold: 标准差倍数阈值（默认5σ）
    
    Returns:
        去除异常值后的信号
    """
    median_val = np.median(data)
    mad = np.median(np.abs(data - median_val))
    
    # 使用MAD估计标准差（更稳健）
    std_estimate = 1.4826 * mad
    
    # 计算限制范围
    upper_limit = median_val + threshold * std_estimate
    lower_limit = median_val - threshold * std_estimate
    
    # 限制极端值（而非删除）
    cleaned = np.clip(data, lower_limit, upper_limit)
    return cleaned
```

### 2.2.4 参数选择

| 参数 | 选择值 | 理由 |
|------|--------|------|
| threshold | 5.0 | 保守策略，仅去除极端异常值，避免损失有效信号 |

---

## 2.3 陷波滤波 (Notch Filter)

### 2.3.1 工频干扰问题

中国电网频率为50Hz，电源线和周围电气设备会在ECG信号中引入工频干扰。

### 2.3.2 陷波器设计

采用**IIR陷波滤波器**，在50Hz处产生深度凹陷，而对其他频率几乎无影响。

**传递函数**：
$$
H(z) = \frac{1 - 2\cos(\omega_0)z^{-1} + z^{-2}}{1 - 2r\cos(\omega_0)z^{-1} + r^2z^{-2}}
$$

其中：
- $\omega_0 = 2\pi f_0 / f_s$ 是归一化陷波频率
- $r = 1 - \Delta\omega/2$，$\Delta\omega$ 是带宽
- $Q$ 是品质因子，$Q = f_0 / \Delta f$

### 2.3.3 算法实现

```python
def notch_filter(data, freq=50.0, Q=30.0, fs=250):
    """
    陷波滤波器 - 去除工频干扰
    
    Args:
        data: 输入信号
        freq: 陷波频率 (Hz)
        Q: 品质因子（越大带宽越窄）
        fs: 采样率
    
    Returns:
        滤波后的信号
    """
    nyq = 0.5 * fs
    w0 = freq / nyq  # 归一化频率
    
    # 使用scipy设计IIR陷波滤波器
    b, a = signal.iirnotch(w0, Q)
    
    # 零相位滤波（前向+后向）
    filtered = signal.filtfilt(b, a, data)
    return filtered
```

### 2.3.4 参数选择

| 参数 | 选择值 | 理由 |
|------|--------|------|
| freq | 50.0 Hz | 中国电网频率 |
| Q | 30.0 | 较高Q值，确保窄带阻止，不影响邻近频率 |

---

## 2.4 带通滤波 (Bandpass Filter)

### 2.4.1 设计目标

ECG信号的有效频率成分：
- **P波、T波**: 0.5-10 Hz
- **QRS波群**: 5-40 Hz

设计带通滤波器同时实现：
1. **高通滤波 (>0.5Hz)**: 去除基线漂移
2. **低通滤波 (<40Hz)**: 去除高频肌电干扰和采样噪声

### 2.4.2 Butterworth滤波器

选择**Butterworth滤波器**的原因：
- 通带内最大平坦（无纹波）
- 相位特性良好
- 设计简单

**幅频特性**：
$$
|H(j\omega)|^2 = \frac{1}{1 + (\omega/\omega_c)^{2n}}
$$

其中 $n$ 是滤波器阶数，$\omega_c$ 是截止频率。

### 2.4.3 算法实现

```python
def bandpass_filter(data, lowcut=0.5, highcut=40.0, order=4, fs=250):
    """
    带通滤波器 - 去除基线漂移和高频噪声
    
    Args:
        data: 输入信号
        lowcut: 低截止频率 (Hz)
        highcut: 高截止频率 (Hz)
        order: 滤波器阶数
        fs: 采样率
    
    Returns:
        滤波后的信号
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyq
    high = highcut / nyq
    
    # 设计Butterworth带通滤波器
    b, a = signal.butter(order, [low, high], btype='band')
    
    # 零相位滤波（消除相位失真）
    filtered = signal.filtfilt(b, a, data)
    return filtered
```

### 2.4.4 零相位滤波的重要性

使用`filtfilt`实现**零相位滤波**：
- 前向滤波一次，再后向滤波一次
- 消除相位延迟
- 等效阶数翻倍
- 保持R峰等关键波形的时间位置精确

### 2.4.5 参数选择

| 参数 | 选择值 | 理由 |
|------|--------|------|
| lowcut | 0.5 Hz | 去除基线漂移，保留P波低频成分 |
| highcut | 40 Hz | 保留QRS波群高频成分，去除肌电干扰 |
| order | 4 | 平衡滤波效果和计算复杂度 |

---

## 2.5 小波去噪 (Wavelet Denoising)

### 2.5.1 小波变换原理

小波变换提供信号的**时频局部化分析**，特别适合处理非平稳的ECG信号。

**离散小波变换 (DWT)**：
$$
W(j,k) = \sum_{n} x(n) \cdot \psi_{j,k}(n)
$$

其中 $\psi_{j,k}(n) = 2^{-j/2}\psi(2^{-j}n - k)$ 是尺度和平移后的小波基函数。

### 2.5.2 多分辨率分析

DWT将信号分解为**近似系数 (cA)** 和**细节系数 (cD)**：

```
原始信号 x(n)
     │
     ├──▶ 低通滤波 ──▶ 下采样 ──▶ cA1 (近似)
     │
     └──▶ 高通滤波 ──▶ 下采样 ──▶ cD1 (细节)

cA1 可继续分解为 cA2 + cD2 ...
```

### 2.5.3 小波基选择

选择**db6 (Daubechies-6)** 小波的理由：
- 与ECG的QRS波形相似性高
- 消失矩为6，能有效表示多项式信号
- 在ECG处理领域广泛验证

### 2.5.4 自适应阈值

采用改进的**VisuShrink**方法计算自适应阈值：

**噪声标准差估计**（使用最高频细节系数）：
$$
\hat{\sigma} = \frac{\text{median}(|cD_n|)}{0.6745}
$$

**通用阈值**：
$$
\lambda = \hat{\sigma} \sqrt{2 \ln N}
$$

**层级自适应阈值**：
$$
\lambda_j = \frac{\lambda}{1 + 0.1 \times j}
$$

其中 $j$ 是分解层级，层级越低（频率越高），阈值越大。

### 2.5.5 软阈值处理

对每层细节系数应用**软阈值**：
$$
\eta_s(w) = \begin{cases}
\text{sign}(w)(|w| - \lambda) & |w| > \lambda \\
0 & |w| \leq \lambda
\end{cases}
$$

软阈值相比硬阈值的优势：
- 输出信号更平滑
- 避免伪Gibbs现象

### 2.5.6 算法实现

```python
def wavelet_denoise(data, wavelet='db6', level=8, threshold_mode='soft'):
    """
    小波去噪 - 使用自适应阈值
    
    Args:
        data: 输入信号
        wavelet: 小波基函数
        level: 分解层数
        threshold_mode: 阈值模式 ('soft' or 'hard')
    
    Returns:
        去噪后的信号
    """
    import pywt
    
    # 小波分解
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # 估计噪声标准差（使用最高频细节系数）
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    
    # 对每层细节系数应用阈值
    denoised_coeffs = [coeffs[0]]  # 保留近似系数
    
    for i, coeff in enumerate(coeffs[1:], 1):
        # 层级自适应阈值
        threshold = sigma * np.sqrt(2 * np.log(len(data))) / (1.0 + 0.1 * i)
        
        if threshold_mode == 'soft':
            denoised = pywt.threshold(coeff, threshold, mode='soft')
        else:
            denoised = pywt.threshold(coeff, threshold, mode='hard')
        
        denoised_coeffs.append(denoised)
    
    # 小波重构
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    
    # 确保长度一致
    if len(denoised_signal) > len(data):
        denoised_signal = denoised_signal[:len(data)]
    
    return denoised_signal
```

### 2.5.7 参数选择

| 参数 | 选择值 | 理由 |
|------|--------|------|
| wavelet | 'db6' | 与ECG形态匹配良好 |
| level | 8 | 250Hz采样率下覆盖0.5Hz以上所有频带 |
| threshold_mode | 'soft' | 输出更平滑，避免伪Gibbs现象 |

---

## 2.6 信噪比评估

### 2.6.1 SNR计算

评估去噪效果使用**信噪比 (SNR)**：

$$
SNR_{dB} = 10 \log_{10}\left(\frac{P_{signal}}{P_{noise}}\right) = 10 \log_{10}\left(\frac{\sum x_{denoised}^2}{\sum (x_{original} - x_{denoised})^2}\right)
$$

### 2.6.2 算法实现

```python
def calculate_snr(original, denoised):
    """
    计算信噪比 (SNR)
    
    Args:
        original: 原始信号
        denoised: 去噪后信号
    
    Returns:
        SNR (dB)
    """
    noise = original - denoised
    signal_power = np.sum(denoised ** 2)
    noise_power = np.sum(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr
```

---

## 2.7 预处理效果验证

### 2.7.1 各阶段信号特性

| 处理阶段 | 信号特性变化 |
|----------|--------------|
| 原始信号 | 含DC偏移、随机噪声、基线漂移、工频干扰 |
| ADC转换后 | 物理单位(mV)，DC偏移已去除 |
| 异常值处理后 | 极端spike被限制 |
| 陷波后 | 50Hz干扰消除 |
| 带通滤波后 | 基线稳定，高频噪声减少 |
| 小波去噪后 | 信号平滑，波形细节保留 |

### 2.7.2 典型SNR提升

经过完整预处理流程，典型SNR提升为 **>15dB**。

---

*[继续第三部分: R峰检测与心拍分割]*
