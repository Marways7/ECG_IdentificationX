#!/usr/bin/env python3
"""
ECG身份识别系统 - 训练脚本
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.data_loader import ECGDataLoader, analyze_data_quality
from models.ecg_classifier import create_model
from models.trainer import ECGTrainer, prepare_data


def plot_training_history(history: dict, save_path: str):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss曲线
    axes[0].plot(history['train_loss'], label='Train Loss', color='#2E86AB')
    axes[0].plot(history['val_loss'], label='Val Loss', color='#E94F37')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy曲线
    axes[1].plot(history['train_acc'], label='Train Acc', color='#2E86AB')
    axes[1].plot(history['val_acc'], label='Val Acc', color='#E94F37')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存至: {save_path}")


def plot_confusion_matrix(conf_matrix: np.ndarray,
                          class_names: list,
                          save_path: str):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # 在格子中显示数值
    thresh = conf_matrix.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if conf_matrix[i, j] > thresh else 'black',
                    fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混淆矩阵已保存至: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='ECG身份识别系统训练')
    parser.add_argument('--data_dir', type=str, default='ECG_data',
                        help='数据目录')
    parser.add_argument('--model_type', type=str, default='fusion',
                        choices=['cnn', 'tdnn', 'mfcc', 'fusion', 'lightweight'],
                        help='模型类型')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / 'figures'
    reports_dir = output_dir / 'reports'
    models_dir = Path('models') / 'saved'

    for d in [figures_dir, reports_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("ECG身份识别系统 - 训练")
    print("="*60)

    # 1. 加载数据
    print("\n[1/5] 加载和预处理数据...")
    loader = ECGDataLoader(args.data_dir)
    beats, mfcc_features, labels, class_names = loader.prepare_dataset()

    print(f"\n数据集统计:")
    print(f"  - 总心拍数: {len(beats)}")
    print(f"  - 心拍长度: {beats.shape[1]}")
    print(f"  - MFCC特征维度: {mfcc_features.shape[1]}")
    print(f"  - 类别数: {len(class_names)}")
    print(f"  - 类别: {class_names}")

    # 2. 划分数据集
    print("\n[2/5] 划分训练集/验证集/测试集...")
    train_loader, val_loader, test_loader = prepare_data(
        beats, mfcc_features, labels,
        test_size=0.2,
        val_size=0.1,
        batch_size=args.batch_size
    )

    print(f"  - 训练集: {len(train_loader.dataset)} 样本")
    print(f"  - 验证集: {len(val_loader.dataset)} 样本")
    print(f"  - 测试集: {len(test_loader.dataset)} 样本")

    # 3. 创建模型
    print(f"\n[3/5] 创建模型 ({args.model_type})...")
    model = create_model(
        model_type=args.model_type,
        beat_length=beats.shape[1],
        mfcc_dim=mfcc_features.shape[1],
        num_classes=len(class_names)
    )

    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 总参数: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")

    # 4. 训练模型
    print(f"\n[4/5] 训练模型 (epochs={args.epochs})...")
    trainer = ECGTrainer(model, save_dir=str(models_dir))

    history = trainer.train(
        train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=15
    )

    # 绘制训练曲线
    plot_training_history(history, str(figures_dir / 'training_history.png'))

    # 5. 评估模型
    print("\n[5/5] 评估模型...")
    results = trainer.evaluate(test_loader, class_names)

    # 绘制混淆矩阵
    conf_matrix = np.array(results['confusion_matrix'])
    plot_confusion_matrix(conf_matrix, class_names,
                          str(figures_dir / 'confusion_matrix.png'))

    # 保存评估结果
    results_file = reports_dir / 'evaluation_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        # 移除numpy数组（已转为list）
        save_results = {k: v for k, v in results.items()
                        if k not in ['predictions', 'true_labels']}
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f"\n评估结果已保存至: {results_file}")

    # 打印最终结果
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"最终测试准确率: {results['accuracy']:.4f}")
    print(f"最终测试F1分数: {results['f1_score']:.4f}")
    print(f"模型已保存至: {models_dir / 'best_model.pth'}")

    return results


if __name__ == '__main__':
    main()
