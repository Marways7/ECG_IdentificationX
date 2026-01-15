"""
AI智能分析模块
使用DeepSeek API进行ECG数据的智能解读
"""

import os
import json
from typing import Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class ECGAIAnalyzer:
    """ECG AI智能分析器"""

    # 系统提示词 - 专业ECG分析专家角色
    SYSTEM_PROMPT = """你是一位专业的心电图(ECG)分析专家和生物医学工程师，具有以下专业背景：
- 精通心电信号处理和分析
- 熟悉心率变异性(HRV)指标的临床意义
- 了解ECG身份识别技术的原理和应用
- 能够用通俗易懂的语言解释专业概念

你的任务是分析用户提供的ECG数据和处理结果，给出专业、准确、易懂的解读。

回复要求：
1. 使用中文回复
2. 结构清晰，使用适当的标题和列表
3. 先给出关键结论，再详细解释
4. 如有异常值，给出可能的原因和建议
5. 语言专业但不晦涩，适合非专业人士理解
6. 回复控制在300-500字左右"""

    def __init__(self):
        """初始化AI分析器"""
        api_key = os.getenv('DEEPSEEK_API_KEY')
        base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')

        if not api_key:
            raise ValueError("未找到DEEPSEEK_API_KEY环境变量")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = "deepseek-chat"

    def analyze_signal_quality(self, signal_stats: Dict) -> str:
        """
        分析信号质量

        Args:
            signal_stats: 信号统计信息
        """
        prompt = f"""请分析以下ECG信号的质量：

**信号统计信息：**
- 信号长度: {signal_stats.get('length', 'N/A')} 样本 ({signal_stats.get('duration', 'N/A')} 秒)
- 采样率: {signal_stats.get('fs', 250)} Hz
- 幅值范围: {signal_stats.get('min_val', 'N/A'):.4f} ~ {signal_stats.get('max_val', 'N/A'):.4f} mV
- 标准差: {signal_stats.get('std', 'N/A'):.4f} mV
- 检测到R峰数量: {signal_stats.get('r_peaks_count', 'N/A')} 个
- 有效心拍数: {signal_stats.get('valid_beats', 'N/A')} 个

请评估：
1. 信号质量等级（优/良/中/差）
2. 信号是否存在明显噪声或伪迹
3. R峰检测的可靠性
4. 数据是否适合用于身份识别分析"""

        return self._call_api(prompt)

    def analyze_hrv_metrics(self, hrv_data: Dict) -> str:
        """
        分析HRV指标

        Args:
            hrv_data: HRV分析结果
        """
        # 构建HRV数据描述
        time_domain = []
        freq_domain = []
        nonlinear = []

        if 'SDNN' in hrv_data:
            time_domain.append(f"SDNN: {hrv_data['SDNN']:.2f} ms")
        if 'RMSSD' in hrv_data:
            time_domain.append(f"RMSSD: {hrv_data['RMSSD']:.2f} ms")
        if 'pNN50' in hrv_data:
            time_domain.append(f"pNN50: {hrv_data['pNN50']:.2f}%")
        if 'Mean_HR' in hrv_data:
            time_domain.append(f"平均心率: {hrv_data['Mean_HR']:.1f} BPM")
        if 'Mean_RR' in hrv_data:
            time_domain.append(f"平均RR间期: {hrv_data['Mean_RR']:.1f} ms")

        if 'LF' in hrv_data:
            freq_domain.append(f"LF功率: {hrv_data['LF']:.2f} ms²")
        if 'HF' in hrv_data:
            freq_domain.append(f"HF功率: {hrv_data['HF']:.2f} ms²")
        if 'LF_HF_ratio' in hrv_data:
            freq_domain.append(f"LF/HF比值: {hrv_data['LF_HF_ratio']:.2f}")

        if 'SD1' in hrv_data:
            nonlinear.append(f"SD1: {hrv_data['SD1']:.2f} ms")
        if 'SD2' in hrv_data:
            nonlinear.append(f"SD2: {hrv_data['SD2']:.2f} ms")
        if 'ApEn' in hrv_data:
            nonlinear.append(f"近似熵: {hrv_data['ApEn']:.4f}")

        prompt = f"""请分析以下心率变异性(HRV)指标：

**时域指标：**
{chr(10).join(['- ' + item for item in time_domain]) if time_domain else '- 无数据'}

**频域指标：**
{chr(10).join(['- ' + item for item in freq_domain]) if freq_domain else '- 无数据'}

**非线性指标：**
{chr(10).join(['- ' + item for item in nonlinear]) if nonlinear else '- 无数据'}

请分析：
1. 整体自主神经功能状态
2. 交感/副交感神经平衡情况
3. 各指标是否在正常范围内
4. 对受试者健康状态的初步判断
5. 这些HRV特征对身份识别的意义"""

        return self._call_api(prompt)

    def analyze_identification_result(self, result: Dict) -> str:
        """
        分析身份识别结果

        Args:
            result: 识别结果
        """
        prompt = f"""请分析以下ECG身份识别结果：

**识别结果：**
- 预测身份: 受试者 {result.get('predicted_subject', 'N/A')}
- 置信度: {result.get('confidence', 0)*100:.1f}%

**各类别概率分布：**
{self._format_probabilities(result.get('probabilities', {}))}

**模型信息：**
- 模型类型: 1D-CNN轻量级网络
- 训练准确率: 98.44%

请分析：
1. 识别结果的可信度评估
2. 概率分布是否合理
3. 是否存在混淆风险
4. 对识别结果的综合判断"""

        return self._call_api(prompt)

    def analyze_model_performance(self, metrics: Dict) -> str:
        """
        分析模型性能

        Args:
            metrics: 模型评估指标
        """
        # 格式化混淆矩阵
        cm = metrics.get('confusion_matrix', [])
        cm_str = self._format_confusion_matrix(cm)

        # 格式化各类别指标
        per_class = metrics.get('per_class_metrics', {})
        class_str = self._format_per_class_metrics(per_class)

        prompt = f"""请分析以下ECG身份识别模型的性能：

**总体指标：**
- 准确率: {metrics.get('accuracy', 0)*100:.2f}%
- 精确率: {metrics.get('precision', 0)*100:.2f}%
- 召回率: {metrics.get('recall', 0)*100:.2f}%
- F1分数: {metrics.get('f1_score', 0)*100:.2f}%
- 最佳验证准确率: {metrics.get('best_val_accuracy', 0)*100:.2f}%

**数据集划分：**
- 训练集: {metrics.get('dataset_split', {}).get('train', 'N/A')} 样本
- 验证集: {metrics.get('dataset_split', {}).get('val', 'N/A')} 样本
- 测试集: {metrics.get('dataset_split', {}).get('test', 'N/A')} 样本

**混淆矩阵：**
{cm_str}

**各类别性能：**
{class_str}

请分析：
1. 模型整体性能评价
2. 是否存在过拟合或欠拟合
3. 哪些类别识别效果最好/最差
4. 混淆矩阵揭示的模式
5. 改进建议"""

        return self._call_api(prompt)

    def general_consultation(self, question: str, context: Optional[Dict] = None) -> str:
        """
        通用ECG咨询

        Args:
            question: 用户问题
            context: 上下文信息
        """
        context_str = ""
        if context:
            context_str = f"\n\n**当前分析上下文：**\n{json.dumps(context, ensure_ascii=False, indent=2)}"

        prompt = f"""用户关于ECG分析的问题：

{question}
{context_str}

请基于你的专业知识回答用户的问题。如果问题涉及具体数据，请结合上下文信息回答。"""

        return self._call_api(prompt)

    def _call_api(self, prompt: str) -> str:
        """调用DeepSeek API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"AI分析服务暂时不可用: {str(e)}"

    def _format_probabilities(self, probs: Dict) -> str:
        """格式化概率分布"""
        if not probs:
            return "- 无数据"
        lines = []
        for subj, prob in sorted(probs.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20)
            lines.append(f"- {subj}: {prob*100:5.1f}% {bar}")
        return "\n".join(lines)

    def _format_confusion_matrix(self, cm: list) -> str:
        """格式化混淆矩阵"""
        if not cm:
            return "无数据"
        subjects = ['A', 'B', 'C', 'D', 'E', 'F']
        header = "     " + "  ".join([f"{s:>3}" for s in subjects])
        lines = [header]
        for i, row in enumerate(cm):
            line = f"{subjects[i]}  " + "  ".join([f"{v:>3}" for v in row])
            lines.append(line)
        return "\n".join(lines)

    def _format_per_class_metrics(self, metrics: Dict) -> str:
        """格式化各类别指标"""
        if not metrics:
            return "无数据"
        lines = []
        for subj, m in metrics.items():
            lines.append(f"- {subj}: 精确率={m.get('precision', 0):.2%}, "
                        f"召回率={m.get('recall', 0):.2%}, "
                        f"F1={m.get('f1', 0):.2%}")
        return "\n".join(lines)
