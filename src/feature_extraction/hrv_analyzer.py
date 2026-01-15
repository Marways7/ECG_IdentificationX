"""
心率变异性(HRV)分析模块
实现时域、频域和非线性分析指标
"""

import numpy as np
from scipy import signal, interpolate
from scipy.stats import entropy
from typing import Dict, Tuple, Optional
import warnings


class HRVAnalyzer:
    """
    心率变异性分析器
    计算时域、频域和非线性HRV指标
    """

    def __init__(self, fs: int = 250):
        """
        初始化HRV分析器

        Args:
            fs: ECG采样率 (Hz)
        """
        self.fs = fs

    def compute_rr_intervals(self, r_peaks: np.ndarray) -> np.ndarray:
        """
        计算RR间期

        Args:
            r_peaks: R峰索引

        Returns:
            RR间期 (毫秒)
        """
        rr_samples = np.diff(r_peaks)
        rr_ms = (rr_samples / self.fs) * 1000  # 转换为毫秒
        return rr_ms

    def analyze(self, r_peaks: np.ndarray) -> Dict[str, float]:
        """
        执行完整的HRV分析

        Args:
            r_peaks: R峰索引

        Returns:
            HRV指标字典
        """
        if len(r_peaks) < 10:
            warnings.warn("R峰数量不足，无法进行可靠的HRV分析")
            return {}

        rr_intervals = self.compute_rr_intervals(r_peaks)

        # 去除异常RR间期
        rr_intervals = self._remove_ectopic_beats(rr_intervals)

        if len(rr_intervals) < 5:
            return {}

        results = {}

        # 时域分析
        time_domain = self.time_domain_analysis(rr_intervals)
        results.update(time_domain)

        # 频域分析
        freq_domain = self.frequency_domain_analysis(rr_intervals, r_peaks)
        results.update(freq_domain)

        # 非线性分析
        nonlinear = self.nonlinear_analysis(rr_intervals)
        results.update(nonlinear)

        return results

    def _remove_ectopic_beats(self, rr_intervals: np.ndarray,
                               threshold: float = 0.2) -> np.ndarray:
        """
        去除异位搏动导致的异常RR间期

        Args:
            rr_intervals: RR间期 (ms)
            threshold: 阈值 (相对于中位数的比例)

        Returns:
            清理后的RR间期
        """
        median_rr = np.median(rr_intervals)
        lower_bound = median_rr * (1 - threshold)
        upper_bound = median_rr * (1 + threshold)

        valid_mask = (rr_intervals >= lower_bound) & (rr_intervals <= upper_bound)
        return rr_intervals[valid_mask]

    def time_domain_analysis(self, rr_intervals: np.ndarray) -> Dict[str, float]:
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

    def frequency_domain_analysis(self, rr_intervals: np.ndarray,
                                   r_peaks: np.ndarray) -> Dict[str, float]:
        """
        频域HRV分析

        Args:
            rr_intervals: RR间期 (ms)
            r_peaks: R峰索引

        Returns:
            频域指标字典
        """
        results = {}

        # 创建均匀采样的RR间期序列
        rr_times = np.cumsum(rr_intervals) / 1000  # 转换为秒
        rr_times = np.insert(rr_times, 0, 0)

        # 插值到4Hz均匀采样
        interp_fs = 4.0
        t_interp = np.arange(0, rr_times[-1], 1/interp_fs)

        if len(t_interp) < 64:
            return results

        # 三次样条插值
        f = interpolate.CubicSpline(rr_times[:-1], rr_intervals)
        rr_interp = f(t_interp)

        # 去趋势
        rr_interp = signal.detrend(rr_interp)

        # Welch功率谱密度估计
        nperseg = min(256, len(rr_interp) // 2)
        freqs, psd = signal.welch(rr_interp, fs=interp_fs, nperseg=nperseg)

        # 频段定义 (Hz)
        vlf_band = (0.003, 0.04)  # 极低频
        lf_band = (0.04, 0.15)    # 低频
        hf_band = (0.15, 0.4)     # 高频

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

        # 归一化功率
        if (lf_power + hf_power) > 0:
            results['LF_norm'] = (lf_power / (lf_power + hf_power)) * 100
            results['HF_norm'] = (hf_power / (lf_power + hf_power)) * 100
            results['LF_HF_ratio'] = lf_power / hf_power if hf_power > 0 else 0
        else:
            results['LF_norm'] = 0
            results['HF_norm'] = 0
            results['LF_HF_ratio'] = 0

        return results

    def nonlinear_analysis(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        非线性HRV分析

        Args:
            rr_intervals: RR间期 (ms)

        Returns:
            非线性指标字典
        """
        results = {}

        # Poincaré分析
        sd1, sd2 = self._poincare_analysis(rr_intervals)
        results['SD1'] = sd1
        results['SD2'] = sd2
        results['SD1_SD2_ratio'] = sd1 / sd2 if sd2 > 0 else 0

        # 近似熵
        results['ApEn'] = self._approximate_entropy(rr_intervals, m=2, r=0.2)

        # 样本熵
        results['SampEn'] = self._sample_entropy(rr_intervals, m=2, r=0.2)

        # DFA (去趋势波动分析)
        alpha1, alpha2 = self._dfa_analysis(rr_intervals)
        results['DFA_alpha1'] = alpha1  # 短期波动
        results['DFA_alpha2'] = alpha2  # 长期波动

        return results

    def _poincare_analysis(self, rr_intervals: np.ndarray) -> Tuple[float, float]:
        """
        Poincaré图分析

        Args:
            rr_intervals: RR间期

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

    def _approximate_entropy(self, data: np.ndarray,
                              m: int = 2,
                              r: float = 0.2) -> float:
        """
        近似熵计算

        Args:
            data: 时间序列
            m: 嵌入维度
            r: 相似度阈值 (相对于标准差)

        Returns:
            近似熵值
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

    def _sample_entropy(self, data: np.ndarray,
                         m: int = 2,
                         r: float = 0.2) -> float:
        """
        样本熵计算

        Args:
            data: 时间序列
            m: 嵌入维度
            r: 相似度阈值

        Returns:
            样本熵值
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

    def _dfa_analysis(self, data: np.ndarray) -> Tuple[float, float]:
        """
        去趋势波动分析 (DFA)

        Args:
            data: 时间序列

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
            # 分段
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

        if len(fluctuations) < 4:
            return 0.0, 0.0

        scales_arr = np.array([f[0] for f in fluctuations])
        fluct_arr = np.array([f[1] for f in fluctuations])

        # 取对数
        log_scales = np.log10(scales_arr)
        log_fluct = np.log10(fluct_arr + 1e-10)

        # 短期 (4-16拍) 和长期 (16+拍) 分开拟合
        short_mask = scales_arr <= 16
        long_mask = scales_arr > 16

        alpha1 = 0.0
        alpha2 = 0.0

        if np.sum(short_mask) >= 2:
            coeffs1 = np.polyfit(log_scales[short_mask], log_fluct[short_mask], 1)
            alpha1 = coeffs1[0]

        if np.sum(long_mask) >= 2:
            coeffs2 = np.polyfit(log_scales[long_mask], log_fluct[long_mask], 1)
            alpha2 = coeffs2[0]

        return alpha1, alpha2


class CardiopulmonaryCoupling:
    """
    心肺耦合(CPC)分析
    分析心率变异性与呼吸之间的耦合关系
    """

    def __init__(self, fs: int = 250):
        self.fs = fs

    def analyze(self, rr_intervals: np.ndarray,
                resp_signal: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        心肺耦合分析

        Args:
            rr_intervals: RR间期 (ms)
            resp_signal: 呼吸信号 (可选)

        Returns:
            CPC指标字典
        """
        results = {}

        # 使用HRV的HF成分估计呼吸相关的变异
        # 计算HF功率占比作为CPC指标

        hrv = HRVAnalyzer(self.fs)

        # 创建伪R峰索引用于频域分析
        r_peaks = np.cumsum(rr_intervals * self.fs / 1000).astype(int)
        r_peaks = np.insert(r_peaks, 0, 0)

        freq_results = hrv.frequency_domain_analysis(rr_intervals, r_peaks)

        if freq_results:
            # 高频功率占比作为耦合强度指标
            results['CPC_HF_ratio'] = freq_results.get('HF_norm', 0)

            # LF/HF比值 - 反映交感/副交感平衡
            results['CPC_LF_HF'] = freq_results.get('LF_HF_ratio', 0)

            # 呼吸窦性心律不齐强度
            results['RSA_power'] = freq_results.get('HF', 0)

        return results
