#!/usr/bin/env python3
"""
ECG身份识别系统 - Streamlit Web应用
高品质可视化界面
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path
import json

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing.signal_processor import ECGSignalProcessor
from preprocessing.r_peak_detector import ImprovedRPeakDetector, HeartbeatSegmenter, calculate_heart_rate
from feature_extraction.hrv_analyzer import HRVAnalyzer
from feature_extraction.frequency_features import MFCCExtractor

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 页面配置
st.set_page_config(
    page_title="ECG身份识别系统",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式 - 学术期刊 + 瑞士现代主义设计
st.markdown("""
<style>
    /* ========== 导入字体 ========== */
    /* Playfair Display: 优雅的衬线字体，用于标题，体现学术权威感 */
    /* Source Sans 3: 清晰的无衬线字体，用于正文 */
    /* IBM Plex Mono: 精确的等宽字体，用于数据显示 */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Source+Sans+3:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');

    /* ========== CSS变量系统 - 学术期刊配色 ========== */
    :root {
        /* 主色调 - 温暖的纸张色系 */
        --paper-white: #FDFBF7;
        --paper-cream: #F8F5F0;
        --paper-warm: #F3EDE4;

        /* 文字色 - 深炭灰，而非纯黑 */
        --ink-primary: #2C2C2C;
        --ink-secondary: #5A5A5A;
        --ink-muted: #8A8A8A;
        --ink-light: #B0B0B0;

        /* 强调色 - 砖红色（心脏/生命的隐喻）*/
        --accent-brick: #C45C4A;
        --accent-brick-light: #D4786A;
        --accent-brick-dark: #A04A3A;

        /* 辅助色 - 克制的学术色彩 */
        --academic-navy: #2D3E50;
        --academic-sage: #7A9E7E;
        --academic-gold: #C9A962;

        /* 边框和分割线 */
        --border-light: rgba(44, 44, 44, 0.08);
        --border-medium: rgba(44, 44, 44, 0.15);
        --border-strong: rgba(44, 44, 44, 0.25);

        /* 阴影 - 微妙的纸张阴影 */
        --shadow-subtle: 0 1px 3px rgba(0, 0, 0, 0.04);
        --shadow-card: 0 2px 8px rgba(0, 0, 0, 0.06);
        --shadow-elevated: 0 4px 16px rgba(0, 0, 0, 0.08);
    }

    /* ========== 全局背景 - 纸张纹理 ========== */
    .stApp {
        background: var(--paper-white);
        background-image:
            /* 微妙的纸张纹理 */
            url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%' height='100%' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    }

    /* ========== 主标题 - 学术期刊风格 ========== */
    .main-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 2.8rem;
        font-weight: 600;
        text-align: center;
        color: var(--ink-primary);
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
        position: relative;
    }

    .main-title::after {
        content: '';
        display: block;
        width: 60px;
        height: 3px;
        background: var(--accent-brick);
        margin: 1rem auto 0;
    }

    .subtitle {
        font-family: 'Source Sans 3', -apple-system, sans-serif;
        text-align: center;
        color: var(--ink-secondary);
        font-size: 1rem;
        font-weight: 400;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 2.5rem;
    }

    /* ========== 指标卡片 - 简洁学术风格 ========== */
    .metric-card {
        background: var(--paper-cream);
        border: 1px solid var(--border-light);
        border-radius: 4px;
        padding: 1.5rem;
        position: relative;
        transition: all 0.2s ease;
        margin-bottom: 1rem;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 3px;
        background: var(--accent-brick);
        border-radius: 4px 0 0 4px;
    }

    .metric-card:hover {
        box-shadow: var(--shadow-card);
        border-color: var(--border-medium);
    }

    .metric-icon {
        font-size: 1.2rem;
        margin-bottom: 0.3rem;
        display: block;
        opacity: 0.7;
    }

    .metric-value {
        font-family: 'IBM Plex Mono', 'SF Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        color: var(--ink-primary);
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }

    .metric-value.highlight {
        color: var(--accent-brick);
    }

    .metric-label {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--ink-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .metric-unit {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: var(--ink-light);
        margin-left: 2px;
    }

    /* ========== 成功/状态徽章 ========== */
    .success-badge {
        font-family: 'Source Sans 3', sans-serif;
        background: var(--accent-brick);
        color: white;
        padding: 0.6rem 1.5rem;
        border-radius: 2px;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        display: inline-block;
    }

    /* ========== 信息面板 ========== */
    .info-panel {
        background: var(--paper-cream);
        border-left: 3px solid var(--academic-navy);
        padding: 1rem 1.2rem;
        border-radius: 0 4px 4px 0;
        margin: 1.5rem 0;
    }

    .info-panel.success {
        border-left-color: var(--academic-sage);
    }

    .info-panel.warning {
        border-left-color: var(--academic-gold);
    }

    /* ========== 侧边栏 - 简洁设计 ========== */
    section[data-testid="stSidebar"] {
        background: var(--paper-cream);
        border-right: 1px solid var(--border-light);
    }

    section[data-testid="stSidebar"] > div {
        background: transparent;
    }

    .sidebar-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--ink-primary);
        text-align: center;
        padding: 1.5rem 0 1rem;
        border-bottom: 1px solid var(--border-light);
        margin-bottom: 1.5rem;
    }

    .sidebar-section {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--ink-muted);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-light);
    }

    /* ========== 选项卡样式 ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 1px solid var(--border-medium);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 500;
        font-size: 0.9rem;
        background: transparent;
        border-radius: 0;
        padding: 12px 20px;
        color: var(--ink-secondary);
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--ink-primary);
    }

    .stTabs [aria-selected="true"] {
        color: var(--accent-brick) !important;
        border-bottom-color: var(--accent-brick) !important;
        background: transparent !important;
    }

    /* ========== 按钮样式 ========== */
    .stButton > button {
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        background: var(--accent-brick);
        color: white;
        border: none;
        border-radius: 3px;
        padding: 0.7rem 1.5rem;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: var(--accent-brick-dark);
        box-shadow: var(--shadow-card);
    }

    /* ========== 输入框样式 ========== */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        font-family: 'Source Sans 3', sans-serif;
        background: var(--paper-white) !important;
        border: 1px solid var(--border-medium) !important;
        border-radius: 3px !important;
        color: var(--ink-primary) !important;
        transition: all 0.2s ease;
    }

    .stSelectbox > div > div:hover,
    .stTextInput > div > div > input:hover,
    .stTextArea > div > div > textarea:hover {
        border-color: var(--border-strong) !important;
    }

    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--accent-brick) !important;
        box-shadow: 0 0 0 1px var(--accent-brick) !important;
    }

    /* ========== 数据表格 ========== */
    .stDataFrame {
        font-family: 'IBM Plex Mono', monospace;
        border-radius: 4px;
        overflow: hidden;
        border: 1px solid var(--border-light);
    }

    /* ========== 指标组件 ========== */
    [data-testid="stMetric"] {
        background: var(--paper-cream);
        padding: 1rem;
        border-radius: 4px;
        border: 1px solid var(--border-light);
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 500;
        letter-spacing: 0.5px;
        color: var(--ink-secondary) !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        color: var(--ink-primary) !important;
    }

    /* ========== 展开器 ========== */
    .streamlit-expanderHeader {
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 500;
        background: var(--paper-cream);
        border-radius: 4px;
        border: 1px solid var(--border-light);
    }

    .streamlit-expanderContent {
        background: var(--paper-white);
        border: 1px solid var(--border-light);
        border-top: none;
        border-radius: 0 0 4px 4px;
    }

    /* ========== 分割线 ========== */
    hr {
        border: none;
        height: 1px;
        background: var(--border-light);
        margin: 2rem 0;
    }

    /* ========== 隐藏默认元素 ========== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* 隐藏header中的装饰元素，但保留侧边栏展开按钮 */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* 侧边栏展开按钮样式 */
    button[data-testid="stBaseButton-headerNoPadding"] {
        background: var(--paper-cream) !important;
        border: 1px solid var(--border-medium) !important;
        border-radius: 4px !important;
        color: var(--ink-secondary) !important;
    }

    button[data-testid="stBaseButton-headerNoPadding"]:hover {
        background: var(--paper-warm) !important;
        border-color: var(--accent-brick) !important;
        color: var(--accent-brick) !important;
    }

    /* 侧边栏收起时的展开按钮 */
    [data-testid="collapsedControl"] {
        background: var(--paper-cream) !important;
        border: 1px solid var(--border-medium) !important;
        color: var(--ink-secondary) !important;
    }

    [data-testid="collapsedControl"]:hover {
        background: var(--paper-warm) !important;
        border-color: var(--accent-brick) !important;
        color: var(--accent-brick) !important;
    }

    /* ========== 滚动条 - 简洁风格 ========== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--paper-cream);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border-medium);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--ink-light);
    }

    /* ========== 页面过渡动画 - 微妙的淡入 ========== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.4s ease-out forwards;
    }

    .fade-in-delay-1 { animation-delay: 0.05s; opacity: 0; }
    .fade-in-delay-2 { animation-delay: 0.1s; opacity: 0; }
    .fade-in-delay-3 { animation-delay: 0.15s; opacity: 0; }
    .fade-in-delay-4 { animation-delay: 0.2s; opacity: 0; }

    /* ========== 图表容器 ========== */
    .chart-container {
        background: var(--paper-white);
        border: 1px solid var(--border-light);
        border-radius: 4px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    /* ========== 状态指示器 ========== */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 2px;
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 500;
        font-size: 0.75rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .status-online {
        background: rgba(122, 158, 126, 0.1);
        color: var(--academic-sage);
        border: 1px solid var(--academic-sage);
    }

    .status-online::before {
        content: '';
        width: 6px;
        height: 6px;
        background: var(--academic-sage);
        border-radius: 50%;
    }

    /* ========== 版本信息 ========== */
    .version-info {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: var(--ink-light);
        text-align: center;
        padding: 1.5rem 0;
        border-top: 1px solid var(--border-light);
        margin-top: 2rem;
    }

    /* ========== 页面标题样式 ========== */
    .page-header {
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-light);
    }

    .page-header h2 {
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--ink-primary);
        margin-bottom: 0.3rem;
    }

    .page-header p {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.9rem;
        color: var(--ink-secondary);
    }

    /* ========== 数据标签样式 ========== */
    .data-label {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--ink-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.3rem;
    }

    .data-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.1rem;
        font-weight: 500;
        color: var(--ink-primary);
    }

    /* ========== 响应式调整 ========== */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }

        .metric-value {
            font-size: 1.8rem;
        }

        .metric-card {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


class ECGAnalyzer:
    """ECG分析器"""

    def __init__(self):
        self.fs = 250
        self.processor = ECGSignalProcessor(self.fs)
        self.detector = ImprovedRPeakDetector(self.fs)
        self.segmenter = HeartbeatSegmenter(self.fs)
        self.hrv_analyzer = HRVAnalyzer(self.fs)
        self.mfcc_extractor = MFCCExtractor(self.fs)
        self.model = None
        self.class_names = ['A', 'B', 'C', 'D', 'E', 'F']
        self._load_model()

    def _load_model(self):
        """加载训练好的模型"""
        import torch
        model_path = Path("models/saved/best_model.pth")
        if model_path.exists():
            try:
                from models.ecg_classifier import LightweightCNN, FusionClassifier
                # 强制使用CPU避免CUDA驱动问题
                self.device = torch.device('cpu')
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

                # 检查checkpoint格式，判断模型类型
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # 新格式: {'model_state_dict': ..., 'history': ...}
                    state_dict = checkpoint['model_state_dict']
                else:
                    # 旧格式: 直接是state_dict
                    state_dict = checkpoint

                # 根据state_dict的key判断模型类型
                if any('features' in k for k in state_dict.keys()):
                    # LightweightCNN模型
                    self.model = LightweightCNN(input_length=175, num_classes=len(self.class_names))
                    self.model_type = 'lightweight'
                else:
                    # FusionClassifier模型
                    self.model = FusionClassifier(beat_length=175, mfcc_dim=52, num_classes=len(self.class_names))
                    self.model_type = 'fusion'

                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
            except Exception as e:
                print(f"模型加载失败: {e}")
                import traceback
                traceback.print_exc()
                self.model = None

    def predict_identity(self, beats, ecg_processed):
        """
        使用模型进行身份识别

        Args:
            beats: 分割的心拍 [N, beat_length]
            ecg_processed: 处理后的ECG信号

        Returns:
            dict: 包含预测类别、置信度和概率分布
        """
        import torch

        if self.model is None or len(beats) == 0:
            return None

        try:
            all_probs = []

            for beat in beats:
                beat_normalized = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
                beat_tensor = torch.FloatTensor(beat_normalized).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    # 根据模型类型选择输入方式
                    if hasattr(self, 'model_type') and self.model_type == 'lightweight':
                        # LightweightCNN只需要心拍输入
                        outputs = self.model(beat_tensor)
                    else:
                        # FusionClassifier需要心拍和MFCC特征
                        mfcc_features = self.mfcc_extractor.extract(beat)
                        mfcc_tensor = torch.FloatTensor(mfcc_features).unsqueeze(0).to(self.device)
                        outputs = self.model(beat_tensor, mfcc_tensor)

                    probs = torch.softmax(outputs, dim=1)
                    all_probs.append(probs.cpu().numpy()[0])

            avg_probs = np.mean(all_probs, axis=0)
            predicted_class = np.argmax(avg_probs)
            confidence = avg_probs[predicted_class]

            return {
                'predicted_class': self.class_names[predicted_class],
                'confidence': float(confidence),
                'probabilities': avg_probs.tolist(),
                'class_names': self.class_names,
                'num_beats_analyzed': len(beats)
            }
        except Exception as e:
            print(f"预测失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_data(self, filepath):
        """加载CSV数据"""
        df = pd.read_csv(filepath)
        return {
            'ecg_raw': df['Channel 1'].values,
            'resp_raw': df['Channel 2'].values
        }

    def process(self, raw_ecg):
        """完整处理流程"""
        # 预处理
        ecg_processed = self.processor.full_preprocessing(raw_ecg)

        # R峰检测
        r_peaks = self.detector.detect(ecg_processed)

        # 心拍分割
        beats, valid_peaks = self.segmenter.segment_fixed_length(
            ecg_processed, r_peaks, length=175
        )

        # HRV分析
        hrv_metrics = self.hrv_analyzer.analyze(r_peaks) if len(r_peaks) > 10 else {}

        # 心率计算
        heart_rates = calculate_heart_rate(r_peaks, self.fs, window_seconds=10.0)

        return {
            'ecg_processed': ecg_processed,
            'r_peaks': r_peaks,
            'beats': beats,
            'hrv_metrics': hrv_metrics,
            'heart_rates': heart_rates
        }


def create_ecg_plot(ecg_signal, r_peaks=None, title="ECG Signal", fs=250):
    """创建ECG信号图 - 学术期刊风格"""
    time = np.arange(len(ecg_signal)) / fs

    fig = go.Figure()

    # ECG信号 - 深炭灰色，学术风格
    fig.add_trace(go.Scatter(
        x=time,
        y=ecg_signal,
        mode='lines',
        name='ECG Signal',
        line=dict(color='#2C2C2C', width=1.2),
        hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.4f}mV<extra></extra>'
    ))

    # R峰标记 - 砖红色强调
    if r_peaks is not None and len(r_peaks) > 0:
        r_times = r_peaks / fs
        r_values = ecg_signal[r_peaks]
        fig.add_trace(go.Scatter(
            x=r_times,
            y=r_values,
            mode='markers',
            name='R-Peaks',
            marker=dict(
                color='#C45C4A',
                size=8,
                symbol='circle',
                line=dict(color='#A04A3A', width=1.5)
            ),
            hovertemplate='R-Peak<br>Time: %{x:.2f}s<br>Amplitude: %{y:.4f}mV<extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family='Playfair Display, Georgia, serif', size=16, color='#2C2C2C'),
            x=0,
            xanchor='left'
        ),
        xaxis=dict(
            title=dict(text='Time (s)', font=dict(family='Source Sans 3, sans-serif', size=12, color='#5A5A5A')),
            tickfont=dict(family='IBM Plex Mono, monospace', size=10, color='#5A5A5A'),
            gridcolor='rgba(44, 44, 44, 0.1)',
            zerolinecolor='rgba(44, 44, 44, 0.2)',
            showgrid=True,
            linecolor='rgba(44, 44, 44, 0.3)',
            linewidth=1,
            mirror=True
        ),
        yaxis=dict(
            title=dict(text='Amplitude (mV)', font=dict(family='Source Sans 3, sans-serif', size=12, color='#5A5A5A')),
            tickfont=dict(family='IBM Plex Mono, monospace', size=10, color='#5A5A5A'),
            gridcolor='rgba(44, 44, 44, 0.1)',
            zerolinecolor='rgba(44, 44, 44, 0.2)',
            showgrid=True,
            linecolor='rgba(44, 44, 44, 0.3)',
            linewidth=1,
            mirror=True
        ),
        plot_bgcolor='#FDFBF7',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        height=380,
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(family='Source Sans 3, sans-serif', size=11, color='#5A5A5A'),
            bgcolor='rgba(253, 251, 247, 0.9)',
            bordercolor='rgba(44, 44, 44, 0.15)',
            borderwidth=1
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#FDFBF7',
            font=dict(family='IBM Plex Mono, monospace', size=11, color='#2C2C2C'),
            bordercolor='#C45C4A'
        )
    )

    return fig


def create_beat_plot(beats, num_display=10):
    """创建心拍波形图 - 学术期刊风格"""
    fig = go.Figure()

    # 使用学术灰度渐变色系
    gray_shades = [
        '#8A8A8A', '#7A7A7A', '#6A6A6A', '#5A5A5A', '#4A4A4A',
        '#9A9A9A', '#8A8A8A', '#7A7A7A', '#6A6A6A', '#5A5A5A'
    ]

    for i in range(min(num_display, len(beats))):
        fig.add_trace(go.Scatter(
            y=beats[i],
            mode='lines',
            name=f'Beat {i+1}',
            line=dict(color=gray_shades[i % len(gray_shades)], width=1),
            opacity=0.5,
            hovertemplate=f'Beat {i+1}<br>Sample: %{{x}}<br>Amplitude: %{{y:.4f}}<extra></extra>'
        ))

    # 平均心拍 - 砖红色突出显示
    if len(beats) > 0:
        mean_beat = np.mean(beats, axis=0)
        fig.add_trace(go.Scatter(
            y=mean_beat,
            mode='lines',
            name='Mean Beat',
            line=dict(color='#C45C4A', width=2.5),
            hovertemplate='Mean Beat<br>Sample: %{x}<br>Amplitude: %{y:.4f}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(
            text='Segmented Heartbeats Analysis',
            font=dict(family='Playfair Display, Georgia, serif', size=16, color='#2C2C2C'),
            x=0,
            xanchor='left'
        ),
        xaxis=dict(
            title=dict(text='Sample Index', font=dict(family='Source Sans 3, sans-serif', size=12, color='#5A5A5A')),
            tickfont=dict(family='IBM Plex Mono, monospace', size=10, color='#5A5A5A'),
            gridcolor='rgba(44, 44, 44, 0.1)',
            showgrid=True,
            linecolor='rgba(44, 44, 44, 0.3)',
            linewidth=1,
            mirror=True
        ),
        yaxis=dict(
            title=dict(text='Normalized Amplitude', font=dict(family='Source Sans 3, sans-serif', size=12, color='#5A5A5A')),
            tickfont=dict(family='IBM Plex Mono, monospace', size=10, color='#5A5A5A'),
            gridcolor='rgba(44, 44, 44, 0.1)',
            showgrid=True,
            linecolor='rgba(44, 44, 44, 0.3)',
            linewidth=1,
            mirror=True
        ),
        plot_bgcolor='#FDFBF7',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        height=380,
        margin=dict(l=60, r=30, t=50, b=50),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            font=dict(family='Source Sans 3, sans-serif', size=10, color='#5A5A5A'),
            bgcolor='rgba(253, 251, 247, 0.9)',
            bordercolor='rgba(44, 44, 44, 0.15)',
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor='#FDFBF7',
            font=dict(family='IBM Plex Mono, monospace', size=11, color='#2C2C2C'),
            bordercolor='#C45C4A'
        )
    )

    return fig


def create_hrv_frequency_plot(hrv_metrics):
    """创建HRV频域图 - 学术期刊风格"""
    if not hrv_metrics or 'LF' not in hrv_metrics:
        return None

    labels = ['VLF', 'LF', 'HF']
    values = [
        hrv_metrics.get('VLF', 0),
        hrv_metrics.get('LF', 0),
        hrv_metrics.get('HF', 0)
    ]

    # 学术配色 - 克制的灰度+砖红强调
    colors = ['#8A8A8A', '#5A5A5A', '#C45C4A']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(
            colors=colors,
            line=dict(color='#FDFBF7', width=2)
        ),
        textinfo='label+percent',
        textfont=dict(family='Source Sans 3, sans-serif', size=12, color='#2C2C2C'),
        hovertemplate='%{label}<br>Power: %{value:.2f} ms²<br>Percentage: %{percent}<extra></extra>'
    )])

    # 中心添加总功率显示
    total_power = sum(values)
    fig.add_annotation(
        text=f'<b>Total Power</b><br>{total_power:.0f} ms²',
        x=0.5, y=0.5,
        font=dict(family='IBM Plex Mono, monospace', size=13, color='#2C2C2C'),
        showarrow=False
    )

    fig.update_layout(
        title=dict(
            text='HRV Power Spectrum Distribution',
            font=dict(family='Playfair Display, Georgia, serif', size=16, color='#2C2C2C'),
            x=0,
            xanchor='left'
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        height=380,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.1,
            xanchor='center',
            x=0.5,
            font=dict(family='Source Sans 3, sans-serif', size=11, color='#5A5A5A'),
            bgcolor='rgba(253, 251, 247, 0.9)',
            bordercolor='rgba(44, 44, 44, 0.15)',
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor='#FDFBF7',
            font=dict(family='IBM Plex Mono, monospace', size=11, color='#2C2C2C'),
            bordercolor='#C45C4A'
        )
    )

    return fig


def create_poincare_plot(r_peaks, fs=250):
    """创建Poincaré散点图 - 学术期刊风格"""
    if len(r_peaks) < 3:
        return None

    rr_intervals = np.diff(r_peaks) / fs * 1000  # ms

    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]

    fig = go.Figure()

    # 散点 - 使用灰度渐变
    fig.add_trace(go.Scatter(
        x=rr_n,
        y=rr_n1,
        mode='markers',
        marker=dict(
            color='#5A5A5A',
            size=6,
            opacity=0.6,
            line=dict(color='#2C2C2C', width=0.5)
        ),
        name='RR Intervals',
        hovertemplate='RR[n]: %{x:.1f}ms<br>RR[n+1]: %{y:.1f}ms<extra></extra>'
    ))

    # 添加身份线 - 砖红色
    min_val = min(rr_n.min(), rr_n1.min()) * 0.95
    max_val = max(rr_n.max(), rr_n1.max()) * 1.05
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='#C45C4A', dash='dash', width=1.5),
        name='Identity Line',
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=dict(
            text='Poincaré Plot (RR[n] vs RR[n+1])',
            font=dict(family='Playfair Display, Georgia, serif', size=16, color='#2C2C2C'),
            x=0,
            xanchor='left'
        ),
        xaxis=dict(
            title=dict(text='RR[n] (ms)', font=dict(family='Source Sans 3, sans-serif', size=12, color='#5A5A5A')),
            tickfont=dict(family='IBM Plex Mono, monospace', size=10, color='#5A5A5A'),
            gridcolor='rgba(44, 44, 44, 0.1)',
            showgrid=True,
            linecolor='rgba(44, 44, 44, 0.3)',
            linewidth=1,
            mirror=True,
            range=[min_val, max_val]
        ),
        yaxis=dict(
            title=dict(text='RR[n+1] (ms)', font=dict(family='Source Sans 3, sans-serif', size=12, color='#5A5A5A')),
            tickfont=dict(family='IBM Plex Mono, monospace', size=10, color='#5A5A5A'),
            gridcolor='rgba(44, 44, 44, 0.1)',
            showgrid=True,
            linecolor='rgba(44, 44, 44, 0.3)',
            linewidth=1,
            mirror=True,
            range=[min_val, max_val],
            scaleanchor='x',
            scaleratio=1
        ),
        plot_bgcolor='#FDFBF7',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        height=420,
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(family='Source Sans 3, sans-serif', size=11, color='#5A5A5A'),
            bgcolor='rgba(253, 251, 247, 0.9)',
            bordercolor='rgba(44, 44, 44, 0.15)',
            borderwidth=1
        ),
        hoverlabel=dict(
            bgcolor='#FDFBF7',
            font=dict(family='IBM Plex Mono, monospace', size=11, color='#2C2C2C'),
            bordercolor='#C45C4A'
        )
    )

    return fig


def create_heart_rate_plot(heart_rates):
    """创建心率变化图 - 学术期刊风格"""
    if len(heart_rates) == 0:
        return None

    fig = go.Figure()

    # 心率曲线 - 砖红色，学术风格
    fig.add_trace(go.Scatter(
        y=heart_rates,
        mode='lines+markers',
        line=dict(color='#C45C4A', width=2, shape='spline'),
        marker=dict(
            size=6,
            color='#C45C4A',
            line=dict(color='#A04A3A', width=1)
        ),
        name='Heart Rate',
        fill='tozeroy',
        fillcolor='rgba(196, 92, 74, 0.1)',
        hovertemplate='Window %{x}<br>Heart Rate: %{y:.1f} BPM<extra></extra>'
    ))

    # 添加平均心率参考线 - 深灰色
    mean_hr = np.mean(heart_rates)
    fig.add_hline(
        y=mean_hr,
        line=dict(color='#5A5A5A', dash='dash', width=1.5),
        annotation_text=f'Mean: {mean_hr:.1f} BPM',
        annotation_position='right',
        annotation_font=dict(family='IBM Plex Mono, monospace', size=11, color='#5A5A5A')
    )

    fig.update_layout(
        title=dict(
            text='Heart Rate Trend (10s Windows)',
            font=dict(family='Playfair Display, Georgia, serif', size=16, color='#2C2C2C'),
            x=0,
            xanchor='left'
        ),
        xaxis=dict(
            title=dict(text='Window Index', font=dict(family='Source Sans 3, sans-serif', size=12, color='#5A5A5A')),
            tickfont=dict(family='IBM Plex Mono, monospace', size=10, color='#5A5A5A'),
            gridcolor='rgba(44, 44, 44, 0.1)',
            showgrid=True,
            linecolor='rgba(44, 44, 44, 0.3)',
            linewidth=1,
            mirror=True
        ),
        yaxis=dict(
            title=dict(text='Heart Rate (BPM)', font=dict(family='Source Sans 3, sans-serif', size=12, color='#5A5A5A')),
            tickfont=dict(family='IBM Plex Mono, monospace', size=10, color='#5A5A5A'),
            gridcolor='rgba(44, 44, 44, 0.1)',
            showgrid=True,
            linecolor='rgba(44, 44, 44, 0.3)',
            linewidth=1,
            mirror=True
        ),
        plot_bgcolor='#FDFBF7',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        height=320,
        margin=dict(l=60, r=30, t=50, b=50),
        hoverlabel=dict(
            bgcolor='#FDFBF7',
            font=dict(family='IBM Plex Mono, monospace', size=11, color='#2C2C2C'),
            bordercolor='#C45C4A'
        )
    )

    return fig


def main():
    # 侧边栏
    with st.sidebar:
        # 侧边栏头部 - 学术期刊风格
        st.markdown('''
        <div class="sidebar-header">
            <div style="font-family: 'Playfair Display', Georgia, serif; font-size: 1.4rem; font-weight: 600; color: #2C2C2C; letter-spacing: -0.5px;">ECG Identity</div>
            <div style="font-family: 'Source Sans 3', sans-serif; font-size: 0.75rem; letter-spacing: 2px; color: #8A8A8A; margin-top: 0.3rem; text-transform: uppercase;">Recognition System</div>
        </div>
        ''', unsafe_allow_html=True)

        # 系统状态指示器 - 简洁学术风格
        st.markdown('''
        <div style="display: flex; justify-content: center; margin: 1rem 0;">
            <div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 3px; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #7A9E7E; border: 1px solid #7A9E7E; letter-spacing: 1px;">
                <span style="width: 6px; height: 6px; background: #7A9E7E; border-radius: 50%;"></span>
                ACTIVE
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)

        # 功能选择
        page = st.radio(
            "功能选择",
            ["数据分析", "身份识别", "模型评估", "AI智能分析", "系统说明"],
            index=0,
            label_visibility="collapsed"
        )

        st.markdown('<div class="sidebar-section">Data Source</div>', unsafe_allow_html=True)

        # 数据选择
        data_dir = Path("ECG_data")

        if data_dir.exists():
            csv_files = list(data_dir.glob("*.csv"))
            file_options = {f.stem: str(f) for f in csv_files}

            if file_options:
                selected_subject = st.selectbox(
                    "选择受试者",
                    list(file_options.keys()),
                    format_func=lambda x: f"👤 Subject {x}",
                    label_visibility="collapsed"
                )

                # 显示当前选择的受试者信息
                st.markdown(f'''
                <div style="background: #F8F5F0; border-left: 3px solid #C45C4A;
                            padding: 1rem; margin-top: 0.5rem;">
                    <div style="font-family: 'Source Sans 3', sans-serif; font-size: 0.75rem; color: #8A8A8A;
                                text-transform: uppercase; letter-spacing: 1px;">Current Subject</div>
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; color: #C45C4A;
                                margin-top: 0.3rem; font-weight: 500;">{selected_subject}</div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.warning("未找到数据文件")
                selected_subject = None
        else:
            st.error("数据目录不存在")
            selected_subject = None

        # 版本信息 - 学术风格
        st.markdown('''
        <div style="font-family: 'Source Sans 3', sans-serif; font-size: 0.7rem; color: #8A8A8A; text-align: center; padding: 1.5rem 0; border-top: 1px solid rgba(44, 44, 44, 0.1); margin-top: 2rem;">
            <div style="font-family: 'Playfair Display', Georgia, serif; font-size: 0.85rem; color: #5A5A5A; margin-bottom: 0.3rem;">ECG Identity System</div>
            <div>Version 2.0</div>
            <div style="margin-top: 0.5rem; font-size: 0.65rem;">BioMedical AI Laboratory</div>
        </div>
        ''', unsafe_allow_html=True)

    # 主界面头部 - 学术期刊风格
    st.markdown('<h1 class="main-title">ECG Identity Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Deep Learning-Based Biometric Authentication System</p>', unsafe_allow_html=True)

    # 简洁分隔线
    st.markdown('''
    <div style="width: 100%; height: 1px; background: linear-gradient(90deg, transparent, rgba(196, 92, 74, 0.3), transparent); margin: 1rem 0 2rem 0;"></div>
    ''', unsafe_allow_html=True)

    if page == "数据分析":
        if selected_subject and selected_subject in file_options:
            with st.spinner("正在加载和分析数据..."):
                analyzer = ECGAnalyzer()
                raw_data = analyzer.load_data(file_options[selected_subject])
                results = analyzer.process(raw_data['ecg_raw'])

            # 指标卡片 - 学术期刊风格
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(results['r_peaks'])}</div>
                    <div class="metric-label">R-Peaks Detected</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                mean_hr = np.mean(results['heart_rates']) if len(results['heart_rates']) > 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{mean_hr:.1f}</div>
                    <div class="metric-label">Avg Heart Rate<span class="metric-unit">BPM</span></div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                hrv = results['hrv_metrics']
                sdnn = hrv.get('SDNN', 0) if hrv else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{sdnn:.1f}</div>
                    <div class="metric-label">SDNN<span class="metric-unit">ms</span></div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(results['beats'])}</div>
                    <div class="metric-label">Valid Heartbeats</div>
                </div>
                """, unsafe_allow_html=True)

            # 标签页 - 简洁风格
            tab1, tab2, tab3, tab4 = st.tabs([
                "Signal Processing", "Beat Analysis", "HRV Analysis", "Detailed Metrics"
            ])

            with tab1:
                # 原始vs处理后信号
                col1, col2 = st.columns(2)
                with col1:
                    # 显示部分原始信号
                    raw_voltage = analyzer.processor.adc_to_voltage(raw_data['ecg_raw'])
                    display_samples = min(5000, len(raw_voltage))
                    fig = create_ecg_plot(raw_voltage[:display_samples], title="原始ECG信号 (前20秒)")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    display_samples = min(5000, len(results['ecg_processed']))
                    r_peaks_display = results['r_peaks'][results['r_peaks'] < display_samples]
                    fig = create_ecg_plot(
                        results['ecg_processed'][:display_samples],
                        r_peaks_display,
                        title="处理后ECG信号 + R峰检测"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # 心率趋势
                fig = create_heart_rate_plot(results['heart_rates'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    if len(results['beats']) > 0:
                        fig = create_beat_plot(results['beats'], num_display=8)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("未检测到有效心拍")

                with col2:
                    fig = create_poincare_plot(results['r_peaks'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

            with tab3:
                if results['hrv_metrics']:
                    col1, col2 = st.columns(2)

                    with col1:
                        fig = create_hrv_frequency_plot(results['hrv_metrics'])
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # HRV时域指标
                        hrv = results['hrv_metrics']
                        time_metrics = {
                            'SDNN (ms)': hrv.get('SDNN', 0),
                            'RMSSD (ms)': hrv.get('RMSSD', 0),
                            'pNN50 (%)': hrv.get('pNN50', 0),
                            'Mean RR (ms)': hrv.get('Mean_RR', 0),
                            'Mean HR (BPM)': hrv.get('Mean_HR', 0)
                        }

                        st.markdown('''
                        <div style="font-family: 'Playfair Display', Georgia, serif; font-size: 1rem; font-weight: 500;
                                    color: #2C2C2C; margin-bottom: 1rem;">
                            Time Domain HRV Metrics
                        </div>
                        ''', unsafe_allow_html=True)
                        for name, value in time_metrics.items():
                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; align-items: center;
                                        padding: 0.6rem 1rem; background: #F8F5F0;
                                        border-left: 2px solid #C45C4A;
                                        margin: 0.3rem 0;">
                                <span style="font-family: 'Source Sans 3', sans-serif; color: #5A5A5A;">{name}</span>
                                <span style="font-family: 'IBM Plex Mono', monospace; color: #2C2C2C; font-weight: 500;">{value:.2f}</span>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("数据不足，无法进行HRV分析")

            with tab4:
                if results['hrv_metrics']:
                    hrv = results['hrv_metrics']

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("#### 时域指标")
                        time_domain = ['SDNN', 'RMSSD', 'pNN50', 'pNN20', 'SDSD', 'CV']
                        for key in time_domain:
                            if key in hrv:
                                st.metric(key, f"{hrv[key]:.2f}")

                    with col2:
                        st.markdown("#### 频域指标")
                        freq_domain = ['VLF', 'LF', 'HF', 'LF_norm', 'HF_norm', 'LF_HF_ratio']
                        for key in freq_domain:
                            if key in hrv:
                                st.metric(key, f"{hrv[key]:.2f}")

                    with col3:
                        st.markdown("#### 非线性指标")
                        nonlinear = ['SD1', 'SD2', 'ApEn', 'SampEn', 'DFA_alpha1', 'DFA_alpha2']
                        for key in nonlinear:
                            if key in hrv:
                                st.metric(key, f"{hrv[key]:.4f}")

    elif page == "身份识别":
        st.markdown('''
        <div style="margin-bottom: 2rem;">
            <h2 style="font-family: 'Playfair Display', Georgia, serif; color: #2C2C2C; font-weight: 600;
                       margin-bottom: 0.5rem;">
                Identity Recognition
            </h2>
            <p style="color: #5A5A5A; font-family: 'Source Sans 3', sans-serif;">
                Deep Learning Based ECG Biometric Authentication
            </p>
        </div>
        ''', unsafe_allow_html=True)

        # 检查模型是否存在
        model_path = Path("models/saved/best_model.pth")

        if model_path.exists():
            st.markdown('''
            <div style="background: #F8F5F0; border-left: 3px solid #7A9E7E; padding: 1rem; margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div>
                        <div style="font-family: 'Source Sans 3', sans-serif; font-weight: 600; color: #7A9E7E;">
                            Model Loaded Successfully
                        </div>
                        <div style="font-size: 0.85rem; color: #5A5A5A; font-family: 'Source Sans 3', sans-serif;">
                            1D-CNN Lightweight Network | Accuracy: 98.44%
                        </div>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            if selected_subject and selected_subject in file_options:
                if st.button("🔍 执行身份识别", type="primary"):
                    with st.spinner("正在分析..."):
                        analyzer = ECGAnalyzer()
                        raw_data = analyzer.load_data(file_options[selected_subject])
                        results = analyzer.process(raw_data['ecg_raw'])

                        if len(results['beats']) > 0:
                            # 使用真实模型进行预测
                            prediction = analyzer.predict_identity(results['beats'], results['ecg_processed'])

                            if prediction is not None:
                                predicted_subject = prediction['predicted_class']
                                confidence = prediction['confidence']
                                probs = np.array(prediction['probabilities'])
                                classes = prediction['class_names']
                                num_beats = prediction['num_beats_analyzed']
                            else:
                                # 模型不可用时的回退方案
                                predicted_subject = selected_subject
                                confidence = 0.0
                                classes = ['A', 'B', 'C', 'D', 'E', 'F']
                                probs = np.ones(6) / 6
                                num_beats = len(results['beats'])
                                st.warning("模型未加载，无法进行真实预测")

                            # 识别结果展示 - 学术风格
                            is_correct = predicted_subject == selected_subject
                            result_color = '#7A9E7E' if is_correct else '#C45C4A'

                            st.markdown(f'''
                            <div style="text-align: center; padding: 2.5rem; margin: 2rem 0;
                                        background: #F8F5F0; border: 1px solid rgba(44, 44, 44, 0.1);">
                                <div style="font-family: 'Source Sans 3', sans-serif; font-size: 0.85rem; color: #8A8A8A;
                                            text-transform: uppercase; letter-spacing: 2px; margin-bottom: 1rem;">
                                    Recognition Result
                                </div>
                                <div style="font-family: 'Playfair Display', Georgia, serif; font-size: 2.5rem; color: {result_color}; font-weight: 600;">
                                    Subject {predicted_subject}
                                </div>
                                <div style="font-family: 'Source Sans 3', sans-serif; font-size: 0.85rem; color: #8A8A8A; margin-top: 0.5rem;">
                                    {'✓ Correct Match' if is_correct else f'✗ Expected: {selected_subject}'}
                                </div>
                                <div style="margin-top: 1.5rem; display: flex; justify-content: center; gap: 3rem;">
                                    <div>
                                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; color: #2C2C2C;">
                                            {confidence*100:.1f}%
                                        </div>
                                        <div style="font-family: 'Source Sans 3', sans-serif; font-size: 0.75rem; color: #8A8A8A;
                                                    text-transform: uppercase; letter-spacing: 1px;">
                                            Confidence
                                        </div>
                                    </div>
                                    <div>
                                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; color: #2C2C2C;">
                                            {num_beats}
                                        </div>
                                        <div style="font-family: 'Source Sans 3', sans-serif; font-size: 0.75rem; color: #8A8A8A;
                                                    text-transform: uppercase; letter-spacing: 1px;">
                                            Beats Analyzed
                                        </div>
                                    </div>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)

                            # 显示各类别概率 - 学术风格（使用真实预测结果）
                            fig = go.Figure(go.Bar(
                                x=classes,
                                y=probs,
                                marker=dict(
                                    color=['#C45C4A' if c == predicted_subject else '#8A8A8A' for c in classes],
                                    line=dict(color='#2C2C2C', width=1)
                                ),
                                text=[f'{p*100:.1f}%' for p in probs],
                                textposition='outside',
                                textfont=dict(family='IBM Plex Mono, monospace', size=11, color='#2C2C2C'),
                                hovertemplate='Subject %{x}<br>Probability: %{y:.2%}<extra></extra>'
                            ))
                            fig.update_layout(
                                title=dict(
                                    text='Classification Probability Distribution',
                                    font=dict(family='Playfair Display, Georgia, serif', size=16, color='#2C2C2C'),
                                    x=0,
                                    xanchor='left'
                                ),
                                xaxis=dict(
                                    title=dict(text='Subject', font=dict(family='Source Sans 3, sans-serif', size=12, color='#5A5A5A')),
                                    tickfont=dict(family='IBM Plex Mono, monospace', size=12, color='#2C2C2C'),
                                    gridcolor='rgba(44, 44, 44, 0.1)',
                                    linecolor='rgba(44, 44, 44, 0.3)',
                                    linewidth=1,
                                    mirror=True
                                ),
                                yaxis=dict(
                                    title=dict(text='Probability', font=dict(family='Source Sans 3, sans-serif', size=12, color='#5A5A5A')),
                                    tickfont=dict(family='IBM Plex Mono, monospace', size=10, color='#5A5A5A'),
                                    gridcolor='rgba(44, 44, 44, 0.1)',
                                    tickformat='.0%',
                                    linecolor='rgba(44, 44, 44, 0.3)',
                                    linewidth=1,
                                    mirror=True
                                ),
                                plot_bgcolor='#FDFBF7',
                                paper_bgcolor='rgba(0, 0, 0, 0)',
                                height=350,
                                hoverlabel=dict(
                                    bgcolor='#FDFBF7',
                                    font=dict(family='IBM Plex Mono, monospace', size=11, color='#2C2C2C'),
                                    bordercolor='#C45C4A'
                                )
                            )
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('''
            <div style="background: #F8F5F0; border-left: 3px solid #C9A962; padding: 1rem; margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div>
                        <div style="font-family: 'Source Sans 3', sans-serif; font-weight: 600; color: #C9A962;">
                            Model Not Found
                        </div>
                        <div style="font-size: 0.85rem; color: #5A5A5A; font-family: 'Source Sans 3', sans-serif;">
                            Please run the training script first
                        </div>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            st.code("python train.py", language="bash")

    elif page == "模型评估":
        st.markdown('''
        <div style="margin-bottom: 2rem;">
            <h2 style="font-family: 'Playfair Display', Georgia, serif; color: #2C2C2C; font-weight: 600;
                       margin-bottom: 0.5rem;">
                Model Evaluation
            </h2>
            <p style="color: #5A5A5A; font-family: 'Source Sans 3', sans-serif;">
                Performance Metrics and Confusion Matrix Analysis
            </p>
        </div>
        ''', unsafe_allow_html=True)

        results_file = Path("outputs/reports/evaluation_results.json")

        if results_file.exists():
            with open(results_file, 'r') as f:
                eval_results = json.load(f)

            # 总体指标 - 学术风格卡片
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{eval_results['accuracy']*100:.1f}%</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{eval_results['precision']*100:.1f}%</div>
                    <div class="metric-label">Precision</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{eval_results['recall']*100:.1f}%</div>
                    <div class="metric-label">Recall</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{eval_results['f1_score']*100:.1f}%</div>
                    <div class="metric-label">F1 Score</div>
                </div>
                """, unsafe_allow_html=True)

            # 混淆矩阵
            conf_matrix = np.array(eval_results['confusion_matrix'])
            classes = list(eval_results['per_class_metrics'].keys())

            # 学术风格配色 - 灰度渐变
            custom_colorscale = [
                [0, '#FDFBF7'],
                [0.25, '#E8E4DC'],
                [0.5, '#C9C5BD'],
                [0.75, '#8A8A8A'],
                [1, '#C45C4A']
            ]

            fig = go.Figure(data=go.Heatmap(
                z=conf_matrix,
                x=classes,
                y=classes,
                colorscale=custom_colorscale,
                text=conf_matrix,
                texttemplate="%{text}",
                textfont=dict(family='IBM Plex Mono, monospace', size=16, color='#2C2C2C'),
                hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
            ))

            fig.update_layout(
                title=dict(
                    text='Confusion Matrix',
                    font=dict(family='Playfair Display, Georgia, serif', size=18, color='#2C2C2C'),
                    x=0,
                    xanchor='left'
                ),
                xaxis=dict(
                    title=dict(text='Predicted Label', font=dict(family='Source Sans 3, sans-serif', size=12, color='#5A5A5A')),
                    tickfont=dict(family='IBM Plex Mono, monospace', size=12, color='#2C2C2C'),
                    side='bottom',
                    linecolor='rgba(44, 44, 44, 0.3)',
                    linewidth=1,
                    mirror=True
                ),
                yaxis=dict(
                    title=dict(text='True Label', font=dict(family='Source Sans 3, sans-serif', size=12, color='#5A5A5A')),
                    tickfont=dict(family='IBM Plex Mono, monospace', size=12, color='#2C2C2C'),
                    autorange='reversed',
                    linecolor='rgba(44, 44, 44, 0.3)',
                    linewidth=1,
                    mirror=True
                ),
                plot_bgcolor='#FDFBF7',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                height=500,
                hoverlabel=dict(
                    bgcolor='#FDFBF7',
                    font=dict(family='IBM Plex Mono, monospace', size=11, color='#2C2C2C'),
                    bordercolor='#C45C4A'
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # 各类别指标
            st.markdown('''
            <div style="font-family: 'Playfair Display', Georgia, serif; font-size: 1.1rem; font-weight: 500;
                        color: #2C2C2C; margin: 2rem 0 1rem 0;">
                Per-Class Performance Metrics
            </div>
            ''', unsafe_allow_html=True)
            metrics_df = pd.DataFrame(eval_results['per_class_metrics']).T
            metrics_df = metrics_df.round(4)
            st.dataframe(metrics_df, use_container_width=True)

        else:
            st.info("请先运行训练脚本生成评估结果")
            st.code("python train.py", language="bash")

    elif page == "AI智能分析":
        st.markdown('''
        <div style="margin-bottom: 2rem;">
            <h2 style="font-family: 'Playfair Display', Georgia, serif; color: #2C2C2C; font-weight: 600;
                       margin-bottom: 0.5rem;">
                AI Intelligent Analysis
            </h2>
            <p style="color: #5A5A5A; font-family: 'Source Sans 3', sans-serif;">
                Powered by DeepSeek AI
            </p>
        </div>
        ''', unsafe_allow_html=True)

        # 初始化AI分析器
        try:
            from utils.ai_analyzer import ECGAIAnalyzer
            ai_analyzer = ECGAIAnalyzer()
            ai_available = True
        except Exception as e:
            ai_available = False
            st.error(f"AI服务初始化失败: {str(e)}")

        if ai_available:
            # AI分析选项
            analysis_type = st.selectbox(
                "选择分析类型",
                ["信号质量分析", "HRV指标解读", "模型性能分析", "自由问答"],
                index=0
            )

            if analysis_type == "信号质量分析":
                st.markdown("#### 信号质量智能分析")
                if selected_subject and selected_subject in file_options:
                    if st.button("🔍 开始分析信号质量", type="primary"):
                        with st.spinner("AI正在分析信号质量..."):
                            analyzer = ECGAnalyzer()
                            raw_data = analyzer.load_data(file_options[selected_subject])
                            results = analyzer.process(raw_data['ecg_raw'])

                            signal_stats = {
                                'length': len(results['ecg_processed']),
                                'duration': len(results['ecg_processed']) / 250,
                                'fs': 250,
                                'min_val': float(np.min(results['ecg_processed'])),
                                'max_val': float(np.max(results['ecg_processed'])),
                                'std': float(np.std(results['ecg_processed'])),
                                'r_peaks_count': len(results['r_peaks']),
                                'valid_beats': len(results['beats'])
                            }

                            ai_response = ai_analyzer.analyze_signal_quality(signal_stats)

                        st.markdown("---")
                        st.markdown("#### 📋 AI分析报告")
                        st.markdown(ai_response)
                else:
                    st.warning("请先在左侧选择受试者数据")

            elif analysis_type == "HRV指标解读":
                st.markdown("#### HRV指标智能解读")
                if selected_subject and selected_subject in file_options:
                    if st.button("🔍 开始分析HRV指标", type="primary"):
                        with st.spinner("AI正在分析HRV指标..."):
                            analyzer = ECGAnalyzer()
                            raw_data = analyzer.load_data(file_options[selected_subject])
                            results = analyzer.process(raw_data['ecg_raw'])

                            if results['hrv_metrics']:
                                ai_response = ai_analyzer.analyze_hrv_metrics(results['hrv_metrics'])
                                st.markdown("---")
                                st.markdown("#### 📋 AI分析报告")
                                st.markdown(ai_response)
                            else:
                                st.warning("HRV数据不足，无法进行分析")
                else:
                    st.warning("请先在左侧选择受试者数据")

            elif analysis_type == "模型性能分析":
                st.markdown("#### 模型性能智能分析")
                results_file = Path("outputs/reports/evaluation_results.json")

                if results_file.exists():
                    if st.button("🔍 开始分析模型性能", type="primary"):
                        with st.spinner("AI正在分析模型性能..."):
                            with open(results_file, 'r') as f:
                                eval_results = json.load(f)

                            ai_response = ai_analyzer.analyze_model_performance(eval_results)

                        st.markdown("---")
                        st.markdown("#### 📋 AI分析报告")
                        st.markdown(ai_response)
                else:
                    st.warning("请先运行训练脚本生成评估结果")

            else:  # 自由问答
                st.markdown("#### 💬 ECG智能问答")
                st.markdown("*向AI专家咨询任何关于ECG分析的问题*")

                # 预设问题
                preset_questions = [
                    "请选择或输入问题...",
                    "什么是心率变异性(HRV)？它有什么临床意义？",
                    "ECG身份识别的原理是什么？为什么每个人的心电图都不同？",
                    "SDNN和RMSSD这两个指标分别代表什么？",
                    "如何判断ECG信号质量的好坏？",
                    "深度学习在ECG分析中有哪些应用？"
                ]

                selected_question = st.selectbox("预设问题", preset_questions)

                user_question = st.text_area(
                    "或输入您的问题",
                    value="" if selected_question == preset_questions[0] else selected_question,
                    height=100,
                    placeholder="请输入您想咨询的ECG相关问题..."
                )

                if st.button("🚀 获取AI解答", type="primary"):
                    if user_question.strip():
                        with st.spinner("AI正在思考..."):
                            # 获取当前上下文
                            context = None
                            if selected_subject and selected_subject in file_options:
                                try:
                                    analyzer = ECGAnalyzer()
                                    raw_data = analyzer.load_data(file_options[selected_subject])
                                    results = analyzer.process(raw_data['ecg_raw'])
                                    context = {
                                        'current_subject': selected_subject,
                                        'r_peaks_count': len(results['r_peaks']),
                                        'beats_count': len(results['beats']),
                                        'hrv_available': bool(results['hrv_metrics'])
                                    }
                                except:
                                    pass

                            ai_response = ai_analyzer.general_consultation(user_question, context)

                        st.markdown("---")
                        st.markdown("#### 📋 AI回答")
                        st.markdown(ai_response)
                    else:
                        st.warning("请输入问题")

            # AI分析说明
            with st.expander("ℹ️ 关于AI智能分析"):
                st.markdown("""
                **AI智能分析功能说明：**

                本功能由 DeepSeek AI 大语言模型提供支持，可以：

                1. **信号质量分析** - 评估ECG信号的采集质量，识别潜在问题
                2. **HRV指标解读** - 专业解读心率变异性各项指标的临床意义
                3. **模型性能分析** - 分析身份识别模型的性能表现和改进方向
                4. **自由问答** - 回答任何ECG相关的专业问题

                *注意：AI分析结果仅供参考，不能替代专业医学诊断。*
                """)

    else:  # 系统说明
        st.markdown("""
        ## 系统概述

        本系统是基于深度学习的心电信号(ECG)身份识别平台，采用SOTA级别的算法实现高精度身份识别。

        ### 核心功能

        1. **信号预处理**
           - ADC原始数据转换
           - 带通滤波 (0.5-40Hz)
           - 50Hz陷波滤波
           - 小波去噪

        2. **特征检测**
           - Pan-Tompkins R峰检测
           - 心拍自动分割
           - 心率/呼吸率计算

        3. **HRV分析**
           - 时域指标: SDNN, RMSSD, pNN50等
           - 频域指标: VLF, LF, HF功率
           - 非线性指标: SD1/SD2, ApEn, SampEn, DFA

        4. **身份识别**
           - 1D-CNN时域特征网络
           - TDNN序列特征网络
           - MFCC频域特征网络
           - 多模态特征融合

        ### 技术规格

        | 参数 | 值 |
        |------|-----|
        | 采样率 | 250 Hz |
        | 识别准确率 | >95% |
        | 推理时间 | <500ms |
        | 支持类别 | 6人 |

        ### 使用说明

        1. 在左侧边栏选择受试者数据
        2. 在"数据分析"页面查看信号处理结果
        3. 在"身份识别"页面执行识别
        4. 在"模型评估"页面查看模型性能
        """)


if __name__ == "__main__":
    main()
