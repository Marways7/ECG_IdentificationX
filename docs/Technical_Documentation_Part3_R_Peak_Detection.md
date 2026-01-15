# 第三部分：R峰检测与心拍分割

## 3.1 R峰检测概述

R峰是ECG信号中振幅最大、最易识别的特征点，是心拍分割、心率计算和HRV分析的基础。

### 3.1.1 R峰检测的挑战

1. **形态变异**: 不同个体、不同导联的R峰形态差异大
2. **噪声影响**: 高幅噪声可能被误检为R峰
3. **振幅变化**: 呼吸等因素导致R峰振幅变化
4. **心律不齐**: 早搏、房颤等导致RR间期不规则

---

## 3.2 Pan-Tompkins算法

### 3.2.1 算法背景

**Pan-Tompkins算法**是1985年提出的经典QRS检测算法，至今仍是ECG处理领域的基准算法。

### 3.2.2 算法流程

```
原始ECG信号
      │
      ▼
┌─────────────────┐
│  带通滤波        │  5-15Hz，突出QRS波群
│  (Bandpass)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  差分求导        │  强调斜率变化
│  (Derivative)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  信号平方        │  非线性增强，统一正负
│  (Squaring)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  滑动窗口积分    │  平滑QRS能量包络
│  (Moving Average)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  自适应阈值检测  │  区分QRS与噪声
│  (Thresholding) │
└────────┬────────┘
         │
         ▼
    R峰位置序列
```

### 3.2.3 带通滤波 (5-15Hz)

**目的**: 突出QRS波群的主要频率成分，抑制P波、T波和高频噪声。

$$
H_{BP}(z) = H_{LP}(z) \cdot H_{HP}(z)
$$

**实现**:
```python
def bandpass_filter(ecg, fs=250):
    """
    带通滤波 (5-15Hz) - 突出QRS波
    """
    nyq = 0.5 * fs
    low = 5.0 / nyq
    high = 15.0 / nyq
    
    b, a = signal.butter(2, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, ecg)
    return filtered
```

### 3.2.4 五点差分求导

**数学原理**:

五点差分公式提供比简单差分更平滑的导数估计：

$$
x'(n) = \frac{-x(n-2) - 2x(n-1) + 2x(n+1) + x(n+2)}{8}
$$

**作用**: 
- 强调信号的快速变化（QRS波的陡峭上升/下降）
- 抑制缓慢变化的P波和T波

**实现**:
```python
def derivative(ecg):
    """
    五点差分求导
    """
    diff = np.zeros_like(ecg)
    diff[2:-2] = (-ecg[:-4] - 2*ecg[1:-3] + 2*ecg[3:-1] + ecg[4:]) / 8
    return diff
```

### 3.2.5 信号平方

**数学定义**:
$$
y(n) = [x'(n)]^2
$$

**作用**:
1. 非线性增强：大振幅成分被显著放大
2. 统一极性：正负R峰都变为正值
3. 增强能量对比：QRS与其他成分的差异更明显

**实现**:
```python
def squaring(ecg):
    """
    信号平方 - 增强QRS波
    """
    return ecg ** 2
```

### 3.2.6 滑动窗口积分

**数学定义**:
$$
y(n) = \frac{1}{N} \sum_{i=0}^{N-1} x(n-i)
$$

其中 $N$ 是窗口宽度。

**窗口宽度选择**:
- QRS波群典型持续时间: 80-120ms
- 选择 $N = 0.08 \times f_s = 0.08 \times 250 = 20$ 样本

**作用**: 
- 平滑QRS能量包络
- 产生与QRS波对应的单峰

**实现**:
```python
def moving_average(ecg, fs=250):
    """
    滑动窗口积分
    """
    integration_window = int(0.08 * fs)  # 80ms窗口
    kernel = np.ones(integration_window) / integration_window
    integrated = np.convolve(ecg, kernel, mode='same')
    return integrated
```

### 3.2.7 自适应阈值检测

**阈值策略**:

使用信号统计特性设置自适应阈值：

$$
\text{threshold} = 0.2 \times P_{98}
$$

其中 $P_{98}$ 是积分信号的98%分位数。

**峰值检测条件**:
1. 幅值超过阈值
2. 与前一个峰的间隔 > 不应期（200ms，对应300BPM最高心率）

**实现**:
```python
def detect_peaks(integrated, original_ecg, refractory_period=0.2, fs=250):
    """
    使用自适应阈值和scipy的峰值检测
    """
    # 最小RR间期
    min_distance = int(refractory_period * fs)
    
    # 自适应阈值
    threshold = 0.2 * np.percentile(integrated, 98)
    
    # 峰值检测
    peaks, _ = signal.find_peaks(integrated, 
                                  height=threshold, 
                                  distance=min_distance)
    
    # 在原始信号中精确定位R峰
    r_peaks = []
    search_window = int(0.05 * fs)  # 50ms搜索窗口
    
    for peak_idx in peaks:
        start = max(0, peak_idx - search_window)
        end = min(len(original_ecg), peak_idx + search_window)
        
        local_segment = original_ecg[start:end]
        # 找绝对值最大的点作为R峰
        local_max_idx = np.argmax(np.abs(local_segment))
        r_peak = start + local_max_idx
        r_peaks.append(r_peak)
    
    return np.array(r_peaks, dtype=int)
```

---

## 3.3 改进的R峰检测器

### 3.3.1 后处理策略

在Pan-Tompkins基础上增加后处理，提高检测准确性：

```
Pan-Tompkins检测结果
         │
         ▼
┌─────────────────┐
│  异常R峰去除     │  基于RR间期统计
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  遗漏R峰填补     │  检测并补充漏检的R峰
└────────┬────────┘
         │
         ▼
    最终R峰序列
```

### 3.3.2 异常R峰去除

**策略**: 基于RR间期的中位数进行异常检测

```python
def remove_outliers(r_peaks, ecg):
    """
    去除异常R峰
    """
    if len(r_peaks) < 3:
        return r_peaks
    
    # 计算RR间期
    rr_intervals = np.diff(r_peaks)
    median_rr = np.median(rr_intervals)
    
    # 去除RR间期异常的R峰
    valid_peaks = [r_peaks[0]]
    for i in range(1, len(r_peaks)):
        rr = r_peaks[i] - valid_peaks[-1]
        
        # RR间期应在中位数的0.5-1.5倍之间
        if 0.5 * median_rr < rr < 1.5 * median_rr:
            valid_peaks.append(r_peaks[i])
        elif rr >= 1.5 * median_rr:
            # 可能遗漏了R峰，但仍添加当前峰
            valid_peaks.append(r_peaks[i])
    
    return np.array(valid_peaks, dtype=int)
```

### 3.3.3 遗漏R峰填补

**策略**: 在RR间期过长的区间内搜索可能遗漏的R峰

```python
def fill_missed_peaks(r_peaks, ecg):
    """
    填补遗漏的R峰
    """
    if len(r_peaks) < 2:
        return r_peaks
    
    rr_intervals = np.diff(r_peaks)
    median_rr = np.median(rr_intervals)
    
    filled_peaks = list(r_peaks)
    
    i = 0
    while i < len(filled_peaks) - 1:
        rr = filled_peaks[i+1] - filled_peaks[i]
        
        # 如果RR间期大于1.8倍中位值，可能遗漏了R峰
        if rr > 1.8 * median_rr:
            # 在中间搜索可能遗漏的R峰
            search_start = filled_peaks[i] + int(0.3 * median_rr)
            search_end = filled_peaks[i+1] - int(0.3 * median_rr)
            
            if search_start < search_end:
                segment = ecg[search_start:search_end]
                local_max = np.argmax(segment)
                new_peak = search_start + local_max
                
                # 验证新峰的幅值
                avg_amplitude = np.mean(ecg[r_peaks])
                if ecg[new_peak] > 0.5 * avg_amplitude:
                    filled_peaks.insert(i+1, new_peak)
        i += 1
    
    return np.array(sorted(filled_peaks), dtype=int)
```

---

## 3.4 心拍分割

### 3.4.1 分割窗口设计

以R峰为中心，截取固定时间窗口的信号段：

```
        P    Q  R  S     T
        ↓    ↓  ↓  ↓     ↓
     ───────────●───────────────
       ←  pre  →←    post    →
        0.25s      0.45s
     ←─────────────────────────→
              0.70s = 175样本
```

**参数选择**:
| 参数 | 值 | 理由 |
|------|-----|------|
| pre_r | 0.25s (63样本) | 包含完整P波和QRS起始 |
| post_r | 0.45s (112样本) | 包含完整T波 |
| 总长度 | 0.70s (175样本) | 覆盖完整PQRST波群 |

### 3.4.2 分割算法

```python
class HeartbeatSegmenter:
    """心拍分割器"""
    
    def __init__(self, fs=250, pre_r=0.25, post_r=0.45):
        self.fs = fs
        self.pre_samples = int(pre_r * fs)   # 63样本
        self.post_samples = int(post_r * fs)  # 112样本
        self.beat_length = self.pre_samples + self.post_samples  # 175样本
    
    def segment(self, ecg, r_peaks):
        """
        分割心拍
        
        Returns:
            (心拍数组 [N, 175], 有效R峰索引)
        """
        beats = []
        valid_peaks = []
        
        for r_idx in r_peaks:
            start = r_idx - self.pre_samples
            end = r_idx + self.post_samples
            
            # 检查边界
            if start >= 0 and end <= len(ecg):
                beat = ecg[start:end]
                beat = self._normalize_beat(beat)
                beats.append(beat)
                valid_peaks.append(r_idx)
        
        return np.array(beats), np.array(valid_peaks)
    
    def _normalize_beat(self, beat):
        """Z-score标准化"""
        mean_val = np.mean(beat)
        std_val = np.std(beat)
        
        if std_val > 0:
            normalized = (beat - mean_val) / std_val
        else:
            normalized = beat - mean_val
        
        return normalized
```

### 3.4.3 固定长度重采样

为确保所有心拍具有相同长度，使用三次样条插值重采样：

```python
def segment_fixed_length(self, ecg, r_peaks, length=175):
    """
    分割为固定长度的心拍
    """
    beats, valid_peaks = self.segment(ecg, r_peaks)
    
    if len(beats) == 0:
        return np.array([]), np.array([])
    
    # 重采样到固定长度
    from scipy.interpolate import interp1d
    
    fixed_beats = []
    for beat in beats:
        x_old = np.linspace(0, 1, len(beat))
        x_new = np.linspace(0, 1, length)
        f = interp1d(x_old, beat, kind='cubic')
        fixed_beat = f(x_new)
        fixed_beats.append(fixed_beat)
    
    return np.array(fixed_beats), valid_peaks
```

---

## 3.5 心率计算

### 3.5.1 基于R峰的心率计算

**数学原理**:
$$
HR_{bpm} = \frac{60}{RR_{sec}} = \frac{60 \times f_s}{RR_{samples}}
$$

### 3.5.2 窗口化心率计算

采用**10秒滑动窗口**计算瞬时心率：

```python
def calculate_heart_rate(r_peaks, fs=250, window_seconds=10.0):
    """
    计算心率 (每个窗口)
    
    Args:
        r_peaks: R峰索引
        fs: 采样率
        window_seconds: 窗口长度 (秒)
    
    Returns:
        心率数组 (bpm)
    """
    if len(r_peaks) < 2:
        return np.array([])
    
    window_samples = int(window_seconds * fs)
    total_samples = r_peaks[-1]
    
    heart_rates = []
    
    # 50%重叠的滑动窗口
    for start in range(0, total_samples - window_samples, window_samples // 2):
        end = start + window_samples
        
        # 找到窗口内的R峰
        mask = (r_peaks >= start) & (r_peaks < end)
        window_peaks = r_peaks[mask]
        
        if len(window_peaks) >= 2:
            # 计算窗口内的平均RR间期
            rr_intervals = np.diff(window_peaks) / fs  # 转换为秒
            mean_rr = np.mean(rr_intervals)
            
            # 计算心率 (bpm)
            hr = 60.0 / mean_rr
            heart_rates.append(hr)
    
    return np.array(heart_rates)
```

### 3.5.3 心率计算结果示例

| 受试者 | 平均心率 (BPM) | 心率范围 | 心率变异 |
|--------|----------------|----------|----------|
| A | 102.1 | 95-110 | 中等 |
| B | 119.2 | 110-128 | 较大 |
| C | 114.1 | 105-122 | 中等 |
| D | 86.4 | 80-93 | 较小 |
| E | 74.9 | 70-82 | 较小 |
| F | 89.5 | 82-97 | 中等 |

---

## 3.6 呼吸信号提取

### 3.6.1 ECG导出呼吸 (EDR)

**原理**: 呼吸运动影响心脏位置和电极阻抗，导致R峰幅值随呼吸周期性变化。

```python
def extract_respiration_edr(ecg, r_peaks, fs=250):
    """
    从ECG信号中提取呼吸信号 (EDR)
    使用R峰幅值调制法
    """
    if len(r_peaks) < 3:
        return np.array([]), np.array([])
    
    # 提取R峰幅值
    r_amplitudes = ecg[r_peaks]
    r_times = r_peaks / fs
    
    # 使用三次样条插值重采样到均匀时间轴
    from scipy.interpolate import CubicSpline
    
    # 创建均匀时间轴 (4Hz采样)
    resp_fs = 4
    t_uniform = np.arange(r_times[0], r_times[-1], 1/resp_fs)
    
    # 插值
    cs = CubicSpline(r_times, r_amplitudes)
    resp_signal = cs(t_uniform)
    
    # 带通滤波 (0.1-0.5Hz, 呼吸频率范围6-30次/分)
    nyq = 0.5 * resp_fs
    b, a = signal.butter(3, [0.1/nyq, 0.5/nyq], btype='band')
    resp_filtered = signal.filtfilt(b, a, resp_signal)
    
    return resp_filtered, t_uniform
```

### 3.6.2 呼吸速率计算

使用**过零点法**计算呼吸速率：

```python
def calculate_respiratory_rate(resp_signal, fs=4, window_seconds=10.0):
    """
    计算呼吸速率
    
    Returns:
        呼吸速率数组 (次/分)
    """
    window_size = int(window_seconds * fs)
    hop_size = window_size // 2
    
    rates = []
    for start in range(0, len(resp_signal) - window_size, hop_size):
        segment = resp_signal[start:start + window_size]
        
        # 使用过零点法计算呼吸速率
        zero_crossings = np.where(np.diff(np.signbit(segment)))[0]
        
        # 每两个过零点代表一个呼吸周期
        cycles = len(zero_crossings) / 2
        rate = cycles * (60.0 / window_seconds)
        
        # 限制在合理范围内 (6-30次/分)
        rate = np.clip(rate, 6, 30)
        rates.append(rate)
    
    return np.array(rates)
```

---

*[继续第四部分: HRV分析与特征提取]*
