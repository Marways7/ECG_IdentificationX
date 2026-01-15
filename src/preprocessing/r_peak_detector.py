"""
R峰检测与心拍分割模块
实现Pan-Tompkins算法和改进版本
"""

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d
from typing import Tuple, List, Optional
import warnings


class PanTompkinsDetector:
    """
    Pan-Tompkins R峰检测算法
    经典且可靠的QRS波检测算法
    """

    def __init__(self, fs: int = 250):
        """
        初始化检测器

        Args:
            fs: 采样率 (Hz)
        """
        self.fs = fs
        self.integration_window = int(0.08 * fs)  # 80ms积分窗口

    def bandpass_filter(self, ecg: np.ndarray) -> np.ndarray:
        """
        带通滤波 (5-15Hz) - 突出QRS波

        Args:
            ecg: ECG信号

        Returns:
            滤波后信号
        """
        nyq = 0.5 * self.fs
        low = 5.0 / nyq
        high = 15.0 / nyq

        b, a = signal.butter(2, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, ecg)
        return filtered

    def derivative(self, ecg: np.ndarray) -> np.ndarray:
        """
        五点差分求导

        Args:
            ecg: 输入信号

        Returns:
            导数信号
        """
        diff = np.zeros_like(ecg)
        diff[2:-2] = (-ecg[:-4] - 2*ecg[1:-3] + 2*ecg[3:-1] + ecg[4:]) / 8
        return diff

    def squaring(self, ecg: np.ndarray) -> np.ndarray:
        """
        信号平方 - 增强QRS波

        Args:
            ecg: 输入信号

        Returns:
            平方后信号
        """
        return ecg ** 2

    def moving_average(self, ecg: np.ndarray) -> np.ndarray:
        """
        滑动窗口积分

        Args:
            ecg: 输入信号

        Returns:
            积分后信号
        """
        kernel = np.ones(self.integration_window) / self.integration_window
        integrated = np.convolve(ecg, kernel, mode='same')
        return integrated

    def detect(self, ecg: np.ndarray,
               refractory_period: float = 0.2) -> np.ndarray:
        """
        检测R峰

        Args:
            ecg: ECG信号
            refractory_period: 不应期 (秒)

        Returns:
            R峰索引数组
        """
        # Pan-Tompkins处理流程
        # 1. 带通滤波
        filtered = self.bandpass_filter(ecg)

        # 2. 求导
        differentiated = self.derivative(filtered)

        # 3. 平方
        squared = self.squaring(differentiated)

        # 4. 滑动窗口积分
        integrated = self.moving_average(squared)

        # 5. 使用scipy的find_peaks进行更可靠的检测
        r_peaks = self._detect_peaks(integrated, ecg, refractory_period)

        return r_peaks

    def _detect_peaks(self, integrated: np.ndarray,
                      original_ecg: np.ndarray,
                      refractory_period: float) -> np.ndarray:
        """
        使用自适应阈值和scipy的峰值检测

        Args:
            integrated: 积分后的信号
            original_ecg: 原始ECG信号
            refractory_period: 不应期

        Returns:
            R峰索引数组
        """
        # 最小RR间期: 不应期对应的采样点数
        # 200ms对应最高心率300BPM，这是生理极限
        min_distance = int(refractory_period * self.fs)

        # 使用更稳健的自适应阈值
        # 基于信号的统计特性而非极值
        threshold = 0.2 * np.percentile(integrated, 98)

        # 使用scipy find_peaks，设置合理的最小距离
        # 正常心率40-200BPM，对应RR间期300-1500ms
        peaks, _ = signal.find_peaks(integrated, height=threshold,
                                      distance=min_distance)

        # 在原始信号中精确定位R峰
        r_peaks = []
        search_window = int(0.05 * self.fs)  # 50ms搜索窗口

        for peak_idx in peaks:
            start = max(0, peak_idx - search_window)
            end = min(len(original_ecg), peak_idx + search_window)

            # 在原始ECG中找最大绝对值（处理正负R峰）
            local_segment = original_ecg[start:end]
            # 找绝对值最大的点作为R峰
            abs_segment = np.abs(local_segment)
            local_max_idx = np.argmax(abs_segment)
            r_peak = start + local_max_idx
            r_peaks.append(r_peak)

        return np.array(r_peaks, dtype=int)

    def _find_local_maxima(self, data: np.ndarray, window: int = 25) -> np.ndarray:
        """
        查找局部极大值

        Args:
            data: 输入信号
            window: 窗口大小

        Returns:
            极大值索引
        """
        local_max = maximum_filter1d(data, size=window)
        maxima = np.where((data == local_max) & (data > 0))[0]
        return maxima


class ImprovedRPeakDetector:
    """
    改进的R峰检测器
    结合多种方法提高检测准确性
    """

    def __init__(self, fs: int = 250):
        self.fs = fs
        self.pan_tompkins = PanTompkinsDetector(fs)

    def detect(self, ecg: np.ndarray) -> np.ndarray:
        """
        改进的R峰检测

        Args:
            ecg: ECG信号

        Returns:
            R峰索引数组
        """
        # 使用Pan-Tompkins检测
        r_peaks = self.pan_tompkins.detect(ecg)

        # 后处理: 去除异常检测
        r_peaks = self._remove_outliers(r_peaks, ecg)

        # 填补遗漏的R峰
        r_peaks = self._fill_missed_peaks(r_peaks, ecg)

        return r_peaks

    def _remove_outliers(self, r_peaks: np.ndarray,
                          ecg: np.ndarray) -> np.ndarray:
        """
        去除异常R峰

        Args:
            r_peaks: R峰索引
            ecg: ECG信号

        Returns:
            清理后的R峰索引
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

    def _fill_missed_peaks(self, r_peaks: np.ndarray,
                            ecg: np.ndarray) -> np.ndarray:
        """
        填补遗漏的R峰

        Args:
            r_peaks: R峰索引
            ecg: ECG信号

        Returns:
            补充后的R峰索引
        """
        if len(r_peaks) < 2:
            return r_peaks

        # 计算中位RR间期
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


class HeartbeatSegmenter:
    """
    心拍分割器
    将ECG信号按R峰分割为单个心拍
    """

    def __init__(self, fs: int = 250,
                 pre_r: float = 0.25,
                 post_r: float = 0.45):
        """
        初始化分割器

        Args:
            fs: 采样率 (Hz)
            pre_r: R峰前的时间窗口 (秒)
            post_r: R峰后的时间窗口 (秒)
        """
        self.fs = fs
        self.pre_samples = int(pre_r * fs)
        self.post_samples = int(post_r * fs)
        self.beat_length = self.pre_samples + self.post_samples

    def segment(self, ecg: np.ndarray,
                r_peaks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        分割心拍

        Args:
            ecg: ECG信号
            r_peaks: R峰索引

        Returns:
            (心拍数组 [N, beat_length], 有效R峰索引)
        """
        beats = []
        valid_peaks = []

        for r_idx in r_peaks:
            start = r_idx - self.pre_samples
            end = r_idx + self.post_samples

            # 检查边界
            if start >= 0 and end <= len(ecg):
                beat = ecg[start:end]

                # 标准化心拍
                beat = self._normalize_beat(beat)

                beats.append(beat)
                valid_peaks.append(r_idx)

        if len(beats) == 0:
            return np.array([]), np.array([])

        return np.array(beats), np.array(valid_peaks)

    def _normalize_beat(self, beat: np.ndarray) -> np.ndarray:
        """
        标准化单个心拍

        Args:
            beat: 心拍信号

        Returns:
            标准化后的心拍
        """
        # Z-score标准化
        mean_val = np.mean(beat)
        std_val = np.std(beat)

        if std_val > 0:
            normalized = (beat - mean_val) / std_val
        else:
            normalized = beat - mean_val

        return normalized

    def segment_fixed_length(self, ecg: np.ndarray,
                              r_peaks: np.ndarray,
                              length: int = 175) -> Tuple[np.ndarray, np.ndarray]:
        """
        分割为固定长度的心拍

        Args:
            ecg: ECG信号
            r_peaks: R峰索引
            length: 目标长度

        Returns:
            (心拍数组 [N, length], 有效R峰索引)
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


def calculate_heart_rate(r_peaks: np.ndarray,
                         fs: int = 250,
                         window_seconds: float = 10.0) -> np.ndarray:
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
