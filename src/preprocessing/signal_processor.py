"""
ECG信号预处理模块
实现ADC转换、滤波去噪、基线漂移去除等功能
"""

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter
import pywt
from typing import Tuple, Optional


class ECGSignalProcessor:
    """ECG信号预处理器"""

    # ADS1292R参数
    VREF = 2.42  # 参考电压 V
    GAIN = 6     # 增益
    ADC_RESOLUTION = 24  # 24位ADC
    SAMPLING_RATE = 250  # Hz

    def __init__(self, fs: int = 250):
        """
        初始化信号处理器

        Args:
            fs: 采样率 (Hz)
        """
        self.fs = fs

    def adc_to_voltage(self, raw_data: np.ndarray) -> np.ndarray:
        """
        ADC原始值转换为电压值(mV)
        公式: result = rawdata × (2 × VREF × 1000) / (GAIN × 2^24)

        Args:
            raw_data: ADC原始数据

        Returns:
            电压值 (mV)
        """
        # 先去除DC偏移 (数据以中心值为基准)
        centered_data = raw_data - np.mean(raw_data)

        # 转换为电压
        voltage = centered_data * (2 * self.VREF * 1000) / (self.GAIN * (2 ** self.ADC_RESOLUTION))
        return voltage

    def remove_edge_samples(self, data: np.ndarray, seconds: float = 1.0) -> np.ndarray:
        """
        去除首尾指定秒数的数据

        Args:
            data: 输入信号
            seconds: 去除的秒数

        Returns:
            裁剪后的信号
        """
        samples_to_remove = int(seconds * self.fs)
        if len(data) > 2 * samples_to_remove:
            return data[samples_to_remove:-samples_to_remove]
        return data

    def bandpass_filter(self, data: np.ndarray,
                        lowcut: float = 0.5,
                        highcut: float = 40.0,
                        order: int = 4) -> np.ndarray:
        """
        带通滤波器 - 去除基线漂移和高频噪声

        Args:
            data: 输入信号
            lowcut: 低截止频率 (Hz)
            highcut: 高截止频率 (Hz)
            order: 滤波器阶数

        Returns:
            滤波后的信号
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq

        # 使用Butterworth滤波器
        b, a = signal.butter(order, [low, high], btype='band')

        # 使用filtfilt实现零相位滤波
        filtered = signal.filtfilt(b, a, data)
        return filtered

    def notch_filter(self, data: np.ndarray,
                     freq: float = 50.0,
                     Q: float = 30.0) -> np.ndarray:
        """
        陷波滤波器 - 去除工频干扰

        Args:
            data: 输入信号
            freq: 陷波频率 (Hz)
            Q: 品质因子

        Returns:
            滤波后的信号
        """
        nyq = 0.5 * self.fs
        w0 = freq / nyq

        b, a = signal.iirnotch(w0, Q)
        filtered = signal.filtfilt(b, a, data)
        return filtered

    def wavelet_denoise(self, data: np.ndarray,
                        wavelet: str = 'db6',
                        level: int = 8,
                        threshold_mode: str = 'soft') -> np.ndarray:
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
        # 小波分解
        coeffs = pywt.wavedec(data, wavelet, level=level)

        # 计算自适应阈值 (使用改进的VisuShrink方法)
        # 估计噪声标准差 (使用最高频细节系数)
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
        elif len(denoised_signal) < len(data):
            denoised_signal = np.pad(denoised_signal, (0, len(data) - len(denoised_signal)))

        return denoised_signal

    def extract_respiration_edr(self, ecg: np.ndarray,
                                 r_peaks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        从ECG信号中提取呼吸信号 (EDR - ECG Derived Respiration)
        使用R峰幅值调制法

        Args:
            ecg: ECG信号
            r_peaks: R峰位置索引

        Returns:
            (呼吸信号, 时间轴)
        """
        if len(r_peaks) < 3:
            return np.array([]), np.array([])

        # 提取R峰幅值
        r_amplitudes = ecg[r_peaks]
        r_times = r_peaks / self.fs

        # 使用三次样条插值重采样到均匀时间轴
        from scipy.interpolate import CubicSpline

        # 创建均匀时间轴 (4Hz采样)
        resp_fs = 4  # 呼吸信号采样率
        t_uniform = np.arange(r_times[0], r_times[-1], 1/resp_fs)

        # 插值
        cs = CubicSpline(r_times, r_amplitudes)
        resp_signal = cs(t_uniform)

        # 带通滤波 (0.1-0.5Hz, 呼吸频率范围6-30次/分)
        nyq = 0.5 * resp_fs
        b, a = signal.butter(3, [0.1/nyq, 0.5/nyq], btype='band')
        resp_filtered = signal.filtfilt(b, a, resp_signal)

        return resp_filtered, t_uniform

    def remove_outliers(self, data: np.ndarray,
                        threshold: float = 5.0) -> np.ndarray:
        """
        去除异常值

        Args:
            data: 输入信号
            threshold: 标准差倍数阈值

        Returns:
            去除异常值后的信号
        """
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))
        # 使用MAD估计标准差 (更稳健)
        std_estimate = 1.4826 * mad

        # 限制极端值
        upper_limit = median_val + threshold * std_estimate
        lower_limit = median_val - threshold * std_estimate

        cleaned = np.clip(data, lower_limit, upper_limit)
        return cleaned

    def full_preprocessing(self, raw_data: np.ndarray,
                           remove_edges: bool = True) -> np.ndarray:
        """
        完整的预处理流程

        Args:
            raw_data: ADC原始数据
            remove_edges: 是否去除首尾数据

        Returns:
            预处理后的ECG信号
        """
        # 1. ADC转换
        voltage = self.adc_to_voltage(raw_data)

        # 2. 去除首尾数据
        if remove_edges:
            voltage = self.remove_edge_samples(voltage, seconds=1.0)

        # 3. 去除异常值 (处理采集中的spike)
        voltage = self.remove_outliers(voltage, threshold=5.0)

        # 4. 陷波滤波去除50Hz工频干扰
        filtered = self.notch_filter(voltage, freq=50.0)

        # 5. 带通滤波 (0.5-40Hz)
        filtered = self.bandpass_filter(filtered, lowcut=0.5, highcut=40.0)

        # 6. 小波去噪
        denoised = self.wavelet_denoise(filtered)

        return denoised

    def calculate_snr(self, original: np.ndarray,
                      denoised: np.ndarray) -> float:
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


class RespirationProcessor:
    """呼吸信号处理器"""

    def __init__(self, fs: int = 250):
        self.fs = fs

    def extract_from_channel(self, resp_channel: np.ndarray) -> np.ndarray:
        """
        从呼吸通道提取呼吸信号

        Args:
            resp_channel: 呼吸通道原始数据

        Returns:
            处理后的呼吸信号
        """
        # 转换为电压
        processor = ECGSignalProcessor(self.fs)
        voltage = processor.adc_to_voltage(resp_channel)

        # 低通滤波 (截止频率1Hz)
        nyq = 0.5 * self.fs
        b, a = signal.butter(4, 1.0/nyq, btype='low')
        filtered = signal.filtfilt(b, a, voltage)

        # 去除趋势
        from scipy.signal import detrend
        detrended = detrend(filtered)

        return detrended

    def calculate_respiratory_rate(self, resp_signal: np.ndarray,
                                    window_seconds: float = 10.0) -> np.ndarray:
        """
        计算呼吸速率

        Args:
            resp_signal: 呼吸信号
            window_seconds: 窗口长度(秒)

        Returns:
            呼吸速率数组 (次/分)
        """
        window_size = int(window_seconds * self.fs)
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
