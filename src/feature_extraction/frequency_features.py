"""
频域特征提取模块
实现MFCC倒谱系数和其他频域特征
"""

import numpy as np
from scipy import signal
from scipy.fftpack import dct
from typing import Tuple, Optional
import warnings


class MFCCExtractor:
    """
    MFCC (梅尔频率倒谱系数) 特征提取器
    用于从心拍信号中提取频域特征
    """

    def __init__(self, fs: int = 250,
                 n_mfcc: int = 13,
                 n_mels: int = 26,
                 n_fft: int = 256,
                 hop_length: int = 64,
                 fmin: float = 0.5,
                 fmax: float = 40.0):
        """
        初始化MFCC提取器

        Args:
            fs: 采样率 (Hz)
            n_mfcc: MFCC系数数量
            n_mels: Mel滤波器数量
            n_fft: FFT点数
            hop_length: 帧移
            fmin: 最低频率
            fmax: 最高频率
        """
        self.fs = fs
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax

        # 创建Mel滤波器组
        self.mel_filters = self._create_mel_filterbank()

    def _hz_to_mel(self, hz: float) -> float:
        """Hz转Mel"""
        return 2595 * np.log10(1 + hz / 700)

    def _mel_to_hz(self, mel: float) -> float:
        """Mel转Hz"""
        return 700 * (10 ** (mel / 2595) - 1)

    def _create_mel_filterbank(self) -> np.ndarray:
        """
        创建Mel滤波器组

        Returns:
            Mel滤波器矩阵 [n_mels, n_fft//2 + 1]
        """
        # Mel频率点
        mel_min = self._hz_to_mel(self.fmin)
        mel_max = self._hz_to_mel(self.fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = np.array([self._mel_to_hz(m) for m in mel_points])

        # 转换为FFT bin索引
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.fs).astype(int)

        # 创建滤波器
        n_freqs = self.n_fft // 2 + 1
        filters = np.zeros((self.n_mels, n_freqs))

        for i in range(self.n_mels):
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

    def extract(self, beat: np.ndarray) -> np.ndarray:
        """
        从单个心拍提取MFCC特征

        Args:
            beat: 心拍信号

        Returns:
            MFCC系数 [n_frames, n_mfcc]
        """
        # 分帧
        frames = self._frame_signal(beat)

        if len(frames) == 0:
            return np.zeros((1, self.n_mfcc))

        # 加窗
        window = np.hamming(frames.shape[1])
        windowed = frames * window

        # FFT
        fft_result = np.fft.rfft(windowed, n=self.n_fft)
        power_spectrum = np.abs(fft_result) ** 2

        # Mel滤波器
        mel_spectrum = np.dot(power_spectrum, self.mel_filters.T)

        # 对数压缩
        log_mel = np.log(mel_spectrum + 1e-10)

        # DCT
        mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :self.n_mfcc]

        return mfcc

    def _frame_signal(self, sig: np.ndarray) -> np.ndarray:
        """
        信号分帧

        Args:
            sig: 输入信号

        Returns:
            帧数组 [n_frames, frame_length]
        """
        frame_length = self.n_fft
        n_frames = (len(sig) - frame_length) // self.hop_length + 1

        if n_frames <= 0:
            return np.array([sig[:min(len(sig), frame_length)]])

        frames = np.zeros((n_frames, frame_length))
        for i in range(n_frames):
            start = i * self.hop_length
            frames[i] = sig[start:start + frame_length]

        return frames

    def extract_with_delta(self, beat: np.ndarray) -> np.ndarray:
        """
        提取MFCC及其一阶和二阶差分

        Args:
            beat: 心拍信号

        Returns:
            MFCC + Delta + Delta-Delta [n_frames, 3 * n_mfcc]
        """
        mfcc = self.extract(beat)

        # 一阶差分 (Delta)
        delta = self._compute_delta(mfcc)

        # 二阶差分 (Delta-Delta)
        delta_delta = self._compute_delta(delta)

        # 拼接
        features = np.concatenate([mfcc, delta, delta_delta], axis=1)

        return features

    def _compute_delta(self, features: np.ndarray, N: int = 2) -> np.ndarray:
        """
        计算差分特征

        Args:
            features: 输入特征
            N: 差分窗口大小

        Returns:
            差分特征
        """
        if features.shape[0] < 2 * N + 1:
            return np.zeros_like(features)

        delta = np.zeros_like(features)
        denominator = 2 * sum([i ** 2 for i in range(1, N + 1)])

        for t in range(N, features.shape[0] - N):
            for n in range(1, N + 1):
                delta[t] += n * (features[t + n] - features[t - n])
            delta[t] /= denominator

        return delta

    def extract_statistics(self, beat: np.ndarray) -> np.ndarray:
        """
        提取MFCC统计特征

        Args:
            beat: 心拍信号

        Returns:
            统计特征向量
        """
        mfcc = self.extract(beat)

        # 计算每个MFCC系数的统计量
        features = []

        # 均值
        features.extend(np.mean(mfcc, axis=0))

        # 标准差
        features.extend(np.std(mfcc, axis=0))

        # 最大值
        features.extend(np.max(mfcc, axis=0))

        # 最小值
        features.extend(np.min(mfcc, axis=0))

        return np.array(features)


class FrequencyDomainFeatures:
    """
    通用频域特征提取器
    """

    def __init__(self, fs: int = 250):
        self.fs = fs

    def extract_psd_features(self, beat: np.ndarray) -> np.ndarray:
        """
        提取功率谱密度特征

        Args:
            beat: 心拍信号

        Returns:
            PSD特征向量
        """
        # 计算PSD
        freqs, psd = signal.welch(beat, fs=self.fs, nperseg=min(64, len(beat)))

        features = []

        # 定义频段
        bands = [
            (0.5, 5, 'VLF'),
            (5, 15, 'LF'),
            (15, 30, 'HF'),
            (30, 40, 'VHF')
        ]

        total_power = np.sum(psd)

        for low, high, name in bands:
            mask = (freqs >= low) & (freqs < high)
            band_power = np.sum(psd[mask]) if np.any(mask) else 0
            features.append(band_power)
            # 相对功率
            features.append(band_power / (total_power + 1e-10))

        # 主频率
        dominant_freq = freqs[np.argmax(psd)]
        features.append(dominant_freq)

        # 频谱质心
        spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
        features.append(spectral_centroid)

        # 频谱带宽
        spectral_bandwidth = np.sqrt(
            np.sum(((freqs - spectral_centroid) ** 2) * psd) / (np.sum(psd) + 1e-10)
        )
        features.append(spectral_bandwidth)

        # 频谱平坦度
        geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
        arithmetic_mean = np.mean(psd)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        features.append(spectral_flatness)

        return np.array(features)

    def extract_wavelet_features(self, beat: np.ndarray,
                                  wavelet: str = 'db4',
                                  level: int = 4) -> np.ndarray:
        """
        提取小波特征

        Args:
            beat: 心拍信号
            wavelet: 小波基
            level: 分解层数

        Returns:
            小波特征向量
        """
        import pywt

        # 小波分解
        coeffs = pywt.wavedec(beat, wavelet, level=level)

        features = []

        for i, coeff in enumerate(coeffs):
            # 能量
            energy = np.sum(coeff ** 2)
            features.append(energy)

            # 均值
            features.append(np.mean(coeff))

            # 标准差
            features.append(np.std(coeff))

            # 熵
            prob = (coeff ** 2) / (np.sum(coeff ** 2) + 1e-10)
            entropy = -np.sum(prob * np.log2(prob + 1e-10))
            features.append(entropy)

        return np.array(features)


class MorphologicalFeatures:
    """
    心拍形态特征提取器
    """

    def __init__(self, fs: int = 250):
        self.fs = fs

    def extract(self, beat: np.ndarray,
                r_location: Optional[int] = None) -> np.ndarray:
        """
        提取形态特征

        Args:
            beat: 心拍信号
            r_location: R峰在心拍中的位置

        Returns:
            形态特征向量
        """
        if r_location is None:
            r_location = len(beat) // 3  # 默认R峰在1/3处

        features = []

        # R峰幅值
        r_amplitude = beat[r_location]
        features.append(r_amplitude)

        # QRS波群特征
        qrs_start = max(0, r_location - int(0.05 * self.fs))
        qrs_end = min(len(beat), r_location + int(0.05 * self.fs))
        qrs_segment = beat[qrs_start:qrs_end]

        # QRS幅值范围
        features.append(np.max(qrs_segment) - np.min(qrs_segment))

        # QRS面积
        features.append(np.trapz(np.abs(qrs_segment)))

        # QRS能量
        features.append(np.sum(qrs_segment ** 2))

        # T波特征 (R峰后0.1-0.3秒)
        t_start = r_location + int(0.1 * self.fs)
        t_end = min(len(beat), r_location + int(0.3 * self.fs))
        if t_end > t_start:
            t_segment = beat[t_start:t_end]
            features.append(np.max(t_segment))  # T波峰值
            features.append(np.trapz(t_segment))  # T波面积
        else:
            features.extend([0, 0])

        # P波特征 (R峰前0.1-0.2秒)
        p_start = max(0, r_location - int(0.2 * self.fs))
        p_end = r_location - int(0.1 * self.fs)
        if p_end > p_start:
            p_segment = beat[p_start:p_end]
            features.append(np.max(p_segment))  # P波峰值
            features.append(np.trapz(p_segment))  # P波面积
        else:
            features.extend([0, 0])

        # 统计特征
        features.append(np.mean(beat))
        features.append(np.std(beat))
        features.append(np.max(beat))
        features.append(np.min(beat))

        # 斜率特征
        diff_beat = np.diff(beat)
        features.append(np.max(np.abs(diff_beat)))  # 最大斜率
        features.append(np.mean(np.abs(diff_beat)))  # 平均斜率

        # 峰值比率
        max_val = np.max(beat)
        min_val = np.min(beat)
        features.append(max_val / (np.abs(min_val) + 1e-10))

        return np.array(features)


def extract_all_features(beat: np.ndarray, fs: int = 250) -> np.ndarray:
    """
    提取所有特征

    Args:
        beat: 心拍信号
        fs: 采样率

    Returns:
        综合特征向量
    """
    mfcc_extractor = MFCCExtractor(fs)
    freq_extractor = FrequencyDomainFeatures(fs)
    morph_extractor = MorphologicalFeatures(fs)

    # MFCC统计特征
    mfcc_features = mfcc_extractor.extract_statistics(beat)

    # PSD特征
    psd_features = freq_extractor.extract_psd_features(beat)

    # 小波特征
    wavelet_features = freq_extractor.extract_wavelet_features(beat)

    # 形态特征
    morph_features = morph_extractor.extract(beat)

    # 拼接所有特征
    all_features = np.concatenate([
        mfcc_features,
        psd_features,
        wavelet_features,
        morph_features
    ])

    return all_features
