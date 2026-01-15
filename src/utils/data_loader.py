"""
数据加载和处理工具
"""

import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.signal_processor import ECGSignalProcessor
from preprocessing.r_peak_detector import (ImprovedRPeakDetector,
                                            HeartbeatSegmenter,
                                            calculate_heart_rate)
from feature_extraction.hrv_analyzer import HRVAnalyzer
from feature_extraction.frequency_features import MFCCExtractor, extract_all_features


class ECGDataLoader:
    """
    ECG数据加载器
    加载和预处理原始CSV数据
    """

    def __init__(self, data_dir: str, fs: int = 250):
        """
        初始化数据加载器

        Args:
            data_dir: 数据目录
            fs: 采样率
        """
        self.data_dir = data_dir
        self.fs = fs
        self.processor = ECGSignalProcessor(fs)
        self.detector = ImprovedRPeakDetector(fs)
        self.segmenter = HeartbeatSegmenter(fs)
        self.hrv_analyzer = HRVAnalyzer(fs)
        self.mfcc_extractor = MFCCExtractor(fs)

    def load_csv(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        加载CSV文件

        Args:
            filepath: CSV文件路径

        Returns:
            包含各通道数据的字典
        """
        df = pd.read_csv(filepath)

        data = {
            'timestamp': df['timestamp'].values,
            'ecg_raw': df['Channel 1'].values,
            'resp_raw': df['Channel 2'].values
        }

        return data

    def load_all_subjects(self) -> Dict[str, Dict]:
        """
        加载所有受试者数据

        Returns:
            {subject_id: {raw_data, processed_ecg, r_peaks, beats, ...}}
        """
        subjects = {}
        csv_files = sorted(Path(self.data_dir).glob('*.csv'))

        for csv_file in csv_files:
            subject_id = csv_file.stem  # 文件名作为ID (A, B, C, ...)
            print(f"加载受试者 {subject_id} 的数据...")

            raw_data = self.load_csv(str(csv_file))

            # 预处理ECG信号
            ecg_processed = self.processor.full_preprocessing(
                raw_data['ecg_raw'], remove_edges=True
            )

            # R峰检测
            r_peaks = self.detector.detect(ecg_processed)

            # 心拍分割
            beats, valid_peaks = self.segmenter.segment_fixed_length(
                ecg_processed, r_peaks, length=175
            )

            subjects[subject_id] = {
                'raw_ecg': raw_data['ecg_raw'],
                'raw_resp': raw_data['resp_raw'],
                'processed_ecg': ecg_processed,
                'r_peaks': r_peaks,
                'beats': beats,
                'valid_peaks': valid_peaks
            }

            print(f"  - 信号长度: {len(ecg_processed)} 样本")
            print(f"  - 检测到R峰: {len(r_peaks)} 个")
            print(f"  - 有效心拍: {len(beats)} 个")

        return subjects

    def extract_features_for_subject(self, subject_data: Dict) -> Dict:
        """
        为单个受试者提取所有特征

        Args:
            subject_data: 受试者数据

        Returns:
            特征字典
        """
        beats = subject_data['beats']
        r_peaks = subject_data['valid_peaks']
        ecg = subject_data['processed_ecg']

        features = {
            'mfcc_features': [],
            'morphological_features': [],
            'hrv_features': None
        }

        # 提取每个心拍的特征
        for beat in beats:
            # MFCC特征
            mfcc_stats = self.mfcc_extractor.extract_statistics(beat)
            features['mfcc_features'].append(mfcc_stats)

            # 形态学特征
            morph = extract_all_features(beat, self.fs)
            features['morphological_features'].append(morph)

        features['mfcc_features'] = np.array(features['mfcc_features'])
        features['morphological_features'] = np.array(features['morphological_features'])

        # HRV分析 (整体)
        if len(r_peaks) > 10:
            features['hrv_features'] = self.hrv_analyzer.analyze(r_peaks)

        # 心率计算
        features['heart_rates'] = calculate_heart_rate(r_peaks, self.fs, window_seconds=10.0)

        return features

    def prepare_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        准备完整的数据集

        Returns:
            (beats, mfcc_features, labels, class_names)
        """
        subjects = self.load_all_subjects()

        all_beats = []
        all_mfcc = []
        all_labels = []
        class_names = sorted(subjects.keys())
        label_map = {name: i for i, name in enumerate(class_names)}

        for subject_id, data in subjects.items():
            features = self.extract_features_for_subject(data)

            n_beats = len(data['beats'])
            all_beats.extend(data['beats'])
            all_mfcc.extend(features['mfcc_features'])
            all_labels.extend([label_map[subject_id]] * n_beats)

            print(f"受试者 {subject_id}: {n_beats} 个心拍")

        return (np.array(all_beats),
                np.array(all_mfcc),
                np.array(all_labels),
                class_names)


def analyze_data_quality(subjects: Dict) -> Dict:
    """
    分析数据质量

    Args:
        subjects: 所有受试者数据

    Returns:
        质量分析报告
    """
    report = {}

    for subject_id, data in subjects.items():
        ecg = data['processed_ecg']
        r_peaks = data['r_peaks']
        beats = data['beats']

        # 计算RR间期
        rr_intervals = np.diff(r_peaks) / 250 * 1000  # ms

        report[subject_id] = {
            'signal_length_seconds': len(ecg) / 250,
            'num_r_peaks': len(r_peaks),
            'num_beats': len(beats),
            'mean_rr_ms': np.mean(rr_intervals) if len(rr_intervals) > 0 else 0,
            'std_rr_ms': np.std(rr_intervals) if len(rr_intervals) > 0 else 0,
            'mean_hr_bpm': 60000 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0,
            'signal_quality': 'Good' if len(beats) > 200 else 'Fair'
        }

    return report
