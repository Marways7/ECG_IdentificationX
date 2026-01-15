# 第四部分：HRV分析与特征提取

## 4.1 心率变异性 (HRV) 概述

### 4.1.1 HRV的生理意义

**心率变异性 (Heart Rate Variability, HRV)** 反映自主神经系统对心率的调节能力：

- **交感神经**: 加速心率，降低HRV
- **副交感神经 (迷走神经)**: 减缓心率，增加HRV

HRV指标不仅用于健康评估，也包含**个体特征信息**，可用于身份识别。

### 4.1.2 RR间期计算

HRV分析基于**RR间期 (RR Intervals)** 序列：

$$
RR_i = \frac{(R_{i+1} - R_i)}{f_s} \times 1000 \quad \text{(毫秒)}
$$

```python
def compute_rr_intervals(r_peaks, fs=250):
    """
    计算RR间期 (毫秒)
    """
    rr_samples = np.diff(r_peaks)
    rr_ms = (rr_samples / fs) * 1000
    return rr_ms
```

### 4.1.3 异位搏动去除

在HRV分析前，需去除**异位搏动**（如早搏）导致的异常RR间期：

```python
def remove_ectopic_beats(rr_intervals, threshold=0.2):
    """
    去除异位搏动导致的异常RR间期
    
    Args:
        threshold: 相对于中位数的比例阈值
    """
    median_rr = np.median(rr_intervals)
    lower_bound = median_rr * (1 - threshold)
    upper_bound = median_rr * (1 + threshold)
    
    valid_mask = (rr_intervals >= lower_bound) & (rr_intervals <= upper_bound)
    return rr_intervals[valid_mask]
```

---

## 4.2 时域HRV分析

### 4.2.1 时域指标汇总

| 指标 | 全称 | 公式/定义 | 生理意义 |
|------|------|-----------|----------|
| **Mean_RR** | 平均RR间期 | $\bar{RR} = \frac{1}{N}\sum RR_i$ | 平均心率倒数 |
| **Mean_HR** | 平均心率 | $60000 / \bar{RR}$ | 心脏基础节律 |
| **SDNN** | RR间期标准差 | $\sqrt{\frac{1}{N-1}\sum(RR_i - \bar{RR})^2}$ | 整体HRV |
| **RMSSD** | 连续差值均方根 | $\sqrt{\frac{1}{N-1}\sum(RR_{i+1} - RR_i)^2}$ | 短期变异性 |
| **pNN50** | NN50百分比 | $\frac{|\Delta RR > 50ms|}{N} \times 100\%$ | 副交感活性 |
| **pNN20** | NN20百分比 | $\frac{|\Delta RR > 20ms|}{N} \times 100\%$ | 短期变异性 |
| **SDSD** | 差值标准差 | $\text{std}(RR_{i+1} - RR_i)$ | 瞬时变异性 |
| **CV** | 变异系数 | $\frac{SDNN}{\bar{RR}} \times 100\%$ | 归一化变异性 |

### 4.2.2 算法实现

```python
def time_domain_analysis(rr_intervals):
    """
    时域HRV分析
    
    Args:
        rr_intervals: RR间期 (ms)
    
    Returns:
        时域指标字典
    """
    results = {}
    
    # 平均RR间期
    results['Mean_RR'] = np.mean(rr_intervals)
    
    # 平均心率
    results['Mean_HR'] = 60000 / results['Mean_RR']
    
    # SDNN: RR间期标准差
    results['SDNN'] = np.std(rr_intervals, ddof=1)
    
    # RMSSD: 相邻RR间期差值的均方根
    diff_rr = np.diff(rr_intervals)
    results['RMSSD'] = np.sqrt(np.mean(diff_rr ** 2))
    
    # pNN50: 相邻RR间期差值>50ms的百分比
    nn50 = np.sum(np.abs(diff_rr) > 50)
    results['pNN50'] = (nn50 / len(diff_rr)) * 100
    
    # pNN20: 相邻RR间期差值>20ms的百分比
    nn20 = np.sum(np.abs(diff_rr) > 20)
    results['pNN20'] = (nn20 / len(diff_rr)) * 100
    
    # SDSD: 相邻RR间期差值的标准差
    results['SDSD'] = np.std(diff_rr, ddof=1)
    
    # CV: 变异系数
    results['CV'] = (results['SDNN'] / results['Mean_RR']) * 100
    
    return results
```

---

## 4.3 频域HRV分析

### 4.3.1 频域分析原理

频域分析将RR间期序列分解为不同频率成分，揭示自主神经系统的调节特性。

### 4.3.2 频段定义

| 频段 | 频率范围 | 生理意义 |
|------|----------|----------|
| **VLF** (极低频) | 0.003-0.04 Hz | 体温调节、激素分泌等 |
| **LF** (低频) | 0.04-0.15 Hz | 交感+副交感混合调节 |
| **HF** (高频) | 0.15-0.4 Hz | 副交感/呼吸窦性心律不齐 |

### 4.3.3 RR间期重采样

RR间期是**非均匀采样**的时间序列，需先插值到均匀时间轴：

```python
def resample_rr(rr_intervals, target_fs=4.0):
    """
    将RR间期插值到均匀采样
    
    Args:
        rr_intervals: RR间期 (ms)
        target_fs: 目标采样率 (Hz)
    """
    from scipy import interpolate
    
    # 累积时间
    rr_times = np.cumsum(rr_intervals) / 1000  # 转换为秒
    rr_times = np.insert(rr_times, 0, 0)
    
    # 均匀时间轴
    t_interp = np.arange(0, rr_times[-1], 1/target_fs)
    
    # 三次样条插值
    f = interpolate.CubicSpline(rr_times[:-1], rr_intervals)
    rr_interp = f(t_interp)
    
    # 去趋势
    rr_interp = signal.detrend(rr_interp)
    
    return rr_interp, t_interp
```

### 4.3.4 功率谱密度估计 (Welch方法)

**Welch方法**通过分段平均降低谱估计方差：

1. 将信号分成重叠的段
2. 每段加窗并计算周期图
3. 对所有周期图取平均

```python
def frequency_domain_analysis(rr_intervals, r_peaks, fs=250):
    """
    频域HRV分析
    """
    results = {}
    
    # 重采样到4Hz
    interp_fs = 4.0
    rr_interp, t_interp = resample_rr(rr_intervals, interp_fs)
    
    if len(t_interp) < 64:
        return results
    
    # Welch功率谱密度估计
    nperseg = min(256, len(rr_interp) // 2)
    freqs, psd = signal.welch(rr_interp, fs=interp_fs, nperseg=nperseg)
    
    # 频段定义 (Hz)
    vlf_band = (0.003, 0.04)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    
    # 计算各频段功率
    vlf_mask = (freqs >= vlf_band[0]) & (freqs < vlf_band[1])
    lf_mask = (freqs >= lf_band[0]) & (freqs < lf_band[1])
    hf_mask = (freqs >= hf_band[0]) & (freqs < hf_band[1])
    
    freq_res = freqs[1] - freqs[0]
    
    vlf_power = np.trapz(psd[vlf_mask], dx=freq_res) if np.any(vlf_mask) else 0
    lf_power = np.trapz(psd[lf_mask], dx=freq_res) if np.any(lf_mask) else 0
    hf_power = np.trapz(psd[hf_mask], dx=freq_res) if np.any(hf_mask) else 0
    
    total_power = vlf_power + lf_power + hf_power
    
    results['VLF'] = vlf_power
    results['LF'] = lf_power
    results['HF'] = hf_power
    results['Total_Power'] = total_power
    
    # 归一化功率 (排除VLF)
    if (lf_power + hf_power) > 0:
        results['LF_norm'] = (lf_power / (lf_power + hf_power)) * 100
        results['HF_norm'] = (hf_power / (lf_power + hf_power)) * 100
        results['LF_HF_ratio'] = lf_power / hf_power if hf_power > 0 else 0
    
    return results
```

### 4.3.5 频域指标汇总

| 指标 | 单位 | 计算方法 | 意义 |
|------|------|----------|------|
| VLF | ms² | 0.003-0.04Hz功率积分 | 长期调节 |
| LF | ms² | 0.04-0.15Hz功率积分 | 交感+副交感 |
| HF | ms² | 0.15-0.4Hz功率积分 | 副交感活性 |
| Total_Power | ms² | VLF+LF+HF | 总变异性 |
| LF_norm | % | LF/(LF+HF)×100 | 归一化LF |
| HF_norm | % | HF/(LF+HF)×100 | 归一化HF |
| LF/HF | - | LF/HF | 交感/副交感平衡 |

---

## 4.4 非线性HRV分析

### 4.4.1 Poincaré图分析

**Poincaré图**是将连续RR间期 $(RR_n, RR_{n+1})$ 绘制的散点图。

**SD1和SD2**定义：
- **SD1** (短轴): $SD1 = \frac{1}{\sqrt{2}} \cdot \text{std}(RR_{n+1} - RR_n)$ — 短期变异性
- **SD2** (长轴): $SD2 = \frac{1}{\sqrt{2}} \cdot \text{std}(RR_{n+1} + RR_n)$ — 长期变异性

```python
def poincare_analysis(rr_intervals):
    """
    Poincaré图分析
    
    Returns:
        (SD1, SD2)
    """
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]
    
    # SD1: 短期变异性
    sd1 = np.std(rr_n1 - rr_n, ddof=1) / np.sqrt(2)
    
    # SD2: 长期变异性
    sd2 = np.std(rr_n1 + rr_n, ddof=1) / np.sqrt(2)
    
    return sd1, sd2
```

### 4.4.2 近似熵 (Approximate Entropy, ApEn)

**近似熵**衡量时间序列的复杂度和规律性，值越小表示序列越规则。

**算法步骤**:
1. 构造m维嵌入向量
2. 计算向量间距离小于r的比例
3. 对m和m+1维分别计算
4. ApEn = φ(m) - φ(m+1)

```python
def approximate_entropy(data, m=2, r=0.2):
    """
    近似熵计算
    
    Args:
        data: 时间序列
        m: 嵌入维度
        r: 相似度阈值 (相对于标准差)
    """
    N = len(data)
    r = r * np.std(data)
    
    def _phi(m):
        x = np.array([data[i:i+m] for i in range(N - m + 1)])
        C = np.zeros(len(x))
        for i in range(len(x)):
            dists = np.max(np.abs(x - x[i]), axis=1)
            C[i] = np.sum(dists <= r) / (N - m + 1)
        return np.mean(np.log(C + 1e-10))
    
    return _phi(m) - _phi(m + 1)
```

### 4.4.3 样本熵 (Sample Entropy, SampEn)

**样本熵**是近似熵的改进版，避免了自匹配问题：

```python
def sample_entropy(data, m=2, r=0.2):
    """
    样本熵计算
    """
    N = len(data)
    r = r * np.std(data)
    
    def _count_matches(templates, r):
        count = 0
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count
    
    # m维模板
    templates_m = np.array([data[i:i+m] for i in range(N - m)])
    B = _count_matches(templates_m, r)
    
    # m+1维模板
    templates_m1 = np.array([data[i:i+m+1] for i in range(N - m - 1)])
    A = _count_matches(templates_m1, r)
    
    if B == 0:
        return 0
    
    return -np.log(A / B) if A > 0 else 0
```

### 4.4.4 去趋势波动分析 (DFA)

**DFA (Detrended Fluctuation Analysis)** 分析信号的长程相关性。

**标度指数α的意义**:
- α ≈ 0.5: 白噪声（无相关性）
- α ≈ 1.0: 1/f噪声（长程相关）
- α ≈ 1.5: 布朗噪声（积分过程）

```python
def dfa_analysis(data):
    """
    去趋势波动分析 (DFA)
    
    Returns:
        (alpha1, alpha2) 短期和长期标度指数
    """
    N = len(data)
    
    # 积分序列
    y = np.cumsum(data - np.mean(data))
    
    # 不同尺度
    scales = np.floor(np.logspace(np.log10(4), np.log10(N//4), 20)).astype(int)
    scales = np.unique(scales)
    
    fluctuations = []
    
    for scale in scales:
        n_segments = N // scale
        if n_segments < 2:
            continue
        
        rms_values = []
        for i in range(n_segments):
            start = i * scale
            end = start + scale
            segment = y[start:end]
            
            # 线性去趋势
            x = np.arange(scale)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            rms = np.sqrt(np.mean((segment - trend) ** 2))
            rms_values.append(rms)
        
        fluctuations.append((scale, np.mean(rms_values)))
    
    # 双对数拟合
    scales_arr = np.array([f[0] for f in fluctuations])
    fluct_arr = np.array([f[1] for f in fluctuations])
    
    log_scales = np.log10(scales_arr)
    log_fluct = np.log10(fluct_arr + 1e-10)
    
    # 短期 (4-16拍) 和长期 (16+拍) 分开拟合
    short_mask = scales_arr <= 16
    long_mask = scales_arr > 16
    
    alpha1, alpha2 = 0.0, 0.0
    
    if np.sum(short_mask) >= 2:
        coeffs1 = np.polyfit(log_scales[short_mask], log_fluct[short_mask], 1)
        alpha1 = coeffs1[0]
    
    if np.sum(long_mask) >= 2:
        coeffs2 = np.polyfit(log_scales[long_mask], log_fluct[long_mask], 1)
        alpha2 = coeffs2[0]
    
    return alpha1, alpha2
```

### 4.4.5 非线性指标汇总

| 指标 | 计算方法 | 意义 |
|------|----------|------|
| SD1 | Poincaré短轴 | 短期变异性，与RMSSD相关 |
| SD2 | Poincaré长轴 | 长期变异性，与SDNN相关 |
| SD1/SD2 | 比值 | 短/长期变异性比例 |
| ApEn | 近似熵 | 序列复杂度 |
| SampEn | 样本熵 | 改进的复杂度指标 |
| DFA_α1 | 短期标度指数 | 短程相关性 |
| DFA_α2 | 长期标度指数 | 长程相关性 |

---

## 4.5 MFCC特征提取

### 4.5.1 MFCC原理

**MFCC (Mel-Frequency Cepstral Coefficients)** 是语音识别领域的经典特征，也适用于ECG心拍分析。

**处理流程**:

```
心拍信号
    │
    ▼
┌────────────────┐
│  分帧 + 加窗    │  Hamming窗
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  FFT           │  频谱分析
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Mel滤波器组    │  模拟人耳感知
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  对数压缩      │  动态范围压缩
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  DCT变换       │  去相关
└───────┬────────┘
        │
        ▼
   MFCC系数
```

### 4.5.2 Mel频率尺度

**Mel频率**模拟人耳对频率的非线性感知：

$$
Mel = 2595 \times \log_{10}\left(1 + \frac{f}{700}\right)
$$

$$
f = 700 \times \left(10^{Mel/2595} - 1\right)
$$

```python
def hz_to_mel(hz):
    """Hz转Mel"""
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    """Mel转Hz"""
    return 700 * (10 ** (mel / 2595) - 1)
```

### 4.5.3 Mel滤波器组

```python
def create_mel_filterbank(fs=250, n_mels=26, n_fft=256, fmin=0.5, fmax=40.0):
    """
    创建Mel滤波器组
    
    Returns:
        Mel滤波器矩阵 [n_mels, n_fft//2 + 1]
    """
    # Mel频率点
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    
    # 转换为FFT bin索引
    bin_points = np.floor((n_fft + 1) * hz_points / fs).astype(int)
    
    # 创建三角滤波器
    n_freqs = n_fft // 2 + 1
    filters = np.zeros((n_mels, n_freqs))
    
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        
        # 上升沿
        for j in range(left, center):
            if j < n_freqs and center > left:
                filters[i, j] = (j - left) / (center - left)
        
        # 下降沿
        for j in range(center, right):
            if j < n_freqs and right > center:
                filters[i, j] = (right - j) / (right - center)
    
    return filters
```

### 4.5.4 完整MFCC提取

```python
class MFCCExtractor:
    """MFCC特征提取器"""
    
    def __init__(self, fs=250, n_mfcc=13, n_mels=26, n_fft=256, hop_length=64):
        self.fs = fs
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mel_filters = create_mel_filterbank(fs, n_mels, n_fft)
    
    def extract(self, beat):
        """
        从单个心拍提取MFCC特征
        
        Returns:
            MFCC系数 [n_frames, n_mfcc]
        """
        from scipy.fftpack import dct
        
        # 分帧
        frames = self._frame_signal(beat)
        
        # 加窗 (Hamming)
        window = np.hamming(frames.shape[1])
        windowed = frames * window
        
        # FFT
        fft_result = np.fft.rfft(windowed, n=self.n_fft)
        power_spectrum = np.abs(fft_result) ** 2
        
        # Mel滤波器
        mel_spectrum = np.dot(power_spectrum, self.mel_filters.T)
        
        # 对数压缩
        log_mel = np.log(mel_spectrum + 1e-10)
        
        # DCT (取前n_mfcc个系数)
        mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :self.n_mfcc]
        
        return mfcc
    
    def extract_statistics(self, beat):
        """
        提取MFCC统计特征
        
        Returns:
            统计特征向量 [4 * n_mfcc]
        """
        mfcc = self.extract(beat)
        
        features = []
        features.extend(np.mean(mfcc, axis=0))  # 均值
        features.extend(np.std(mfcc, axis=0))   # 标准差
        features.extend(np.max(mfcc, axis=0))   # 最大值
        features.extend(np.min(mfcc, axis=0))   # 最小值
        
        return np.array(features)
```

### 4.5.5 MFCC参数选择

| 参数 | 选择值 | 理由 |
|------|--------|------|
| n_mfcc | 13 | 经典选择，平衡信息量和维度 |
| n_mels | 26 | 足够的频率分辨率 |
| n_fft | 256 | 约1秒@250Hz，满足频率分辨率 |
| hop_length | 64 | 75%重叠，良好的时间分辨率 |
| fmin | 0.5 Hz | ECG有效频率下限 |
| fmax | 40.0 Hz | ECG有效频率上限 |

---

## 4.6 特征融合策略

### 4.6.1 多模态特征

本系统提取三类特征：

| 特征类型 | 来源 | 维度 | 信息内容 |
|----------|------|------|----------|
| **原始心拍** | 175采样点时序 | 175 | 波形形态 |
| **MFCC统计** | 13MFCC×4统计量 | 52 | 频域特征 |
| **HRV指标** | 时域+频域+非线性 | ~20 | 节律特征 |

### 4.6.2 特征标准化

所有特征在送入分类器前进行**Z-score标准化**：

$$
x_{norm} = \frac{x - \mu}{\sigma}
$$

---

*[继续第五部分: 深度学习模型架构]*
