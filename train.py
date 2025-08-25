#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗问答模型训练脚本
Medical QA Model Training Script

用于训练和管理医疗问答模型的命令行工具
Command line tool for training and managing medical QA models
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from api.utils.training_data_manager import TrainingDataManager
    from api.models.qa_trainer import MedicalQATrainer, ModelConfig, create_default_trainer
    from api.utils.model_evaluator import MedicalQAEvaluator
    from api.utils.auto_learning import AutoLearningManager, AutoLearningConfig
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保在项目根目录下运行此脚本")
    sys.exit(1)

def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log', encoding='utf-8')
        ]
    )

def load_training_config(config_path: str) -> Dict[str, Any]:
    """加载训练配置"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # 返回默认配置
        return {
            "model_name": "hfl/chinese-bert-wwm-ext",
            "model_type": "seq2seq",
            "max_length": 512,
            "max_target_length": 128,
            "learning_rate": 2e-5,
            "batch_size": 4,
            "num_epochs": 3,
            "output_dir": "models/medical_qa"
        }

def prepare_data(args):
    """准备训练数据"""
    print("正在准备训练数据...")
    
    data_manager = TrainingDataManager(args.data_dir)
    
    # 生成问答对
    qa_pairs = data_manager.generate_qa_pairs_from_knowledge_base()
    print(f"从知识库生成了 {len(qa_pairs)} 个问答对")
    
    # 准备训练数据
    train_data, val_data = data_manager.prepare_training_data(
        include_feedback=args.include_feedback,
        train_ratio=args.train_ratio
    )
    
    # 保存数据
    data_manager.save_training_data(train_data, val_data, args.output_dir)
    
    # 显示统计信息
    stats = data_manager.get_data_statistics()
    print("数据统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"训练数据已保存到: {args.output_dir}")

def train_model(args):
    """训练模型"""
    print("开始训练模型...")
    
    # 加载训练配置
    config_dict = load_training_config(args.config)
    
    # 创建模型配置
    model_config = ModelConfig(
        model_name=config_dict.get("model_name", "hfl/chinese-bert-wwm-ext"),
        model_type=config_dict.get("model_type", "seq2seq"),
        max_length=config_dict.get("max_length", 512),
        max_target_length=config_dict.get("max_target_length", 128),
        learning_rate=config_dict.get("learning_rate", 2e-5),
        batch_size=config_dict.get("batch_size", 4),
        num_epochs=config_dict.get("num_epochs", 3),
        output_dir=args.output_dir or config_dict.get("output_dir", "models/medical_qa")
    )
    
    # 创建训练器
    trainer = MedicalQATrainer(model_config)
    
    # 准备数据
    data_manager = TrainingDataManager(args.data_dir)
    train_data, val_data = data_manager.prepare_training_data(
        include_feedback=args.include_feedback,
        train_ratio=args.train_ratio
    )
    
    if len(train_data) < 10:
        print("警告: 训练数据不足，建议至少有10个样本")
        if not args.force:
            print("使用 --force 参数强制训练")
            return
    
    print(f"训练数据: {len(train_data)} 样本")
    print(f"验证数据: {len(val_data)} 样本")
    
    # 开始训练
    try:
        train_result = trainer.train(
            train_data, 
            val_data, 
            resume_from_checkpoint=args.resume_from_checkpoint
        )
        
        print("模型训练完成!")
        print(f"训练时间: {train_result.get('train_runtime', 0):.2f} 秒")
        print(f"训练损失: {train_result.get('train_loss', 0):.4f}")
        print(f"模型保存到: {model_config.output_dir}")
        
    except Exception as e:
        print(f"训练失败: {str(e)}")
        logging.error(f"训练失败: {str(e)}", exc_info=True)

def evaluate_model(args):
    """评估模型"""
    print("开始评估模型...")
    
    if not os.path.exists(args.model_path):
        print(f"模型路径不存在: {args.model_path}")
        return
    
    # 加载模型
    trainer = MedicalQATrainer()
    trainer.load_trained_model(args.model_path)
    
    # 准备评估数据
    data_manager = TrainingDataManager(args.data_dir)
    _, eval_data = data_manager.prepare_training_data(train_ratio=0.8)
    
    # 如果指定了测试文件，使用测试文件
    if args.test_file:
        with open(args.test_file, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
    
    # 生成预测
    predictions = []
    references = []
    
    print(f"评估 {len(eval_data)} 个样本...")
    
    for i, item in enumerate(eval_data):
        if i % 10 == 0:
            print(f"进度: {i}/{len(eval_data)}")
            
        question = item['question']
        context = item.get('context', '')
        reference = item['answer']
        
        try:
            prediction = trainer.predict(question, context)
            predictions.append(prediction)
            references.append(reference)
        except Exception as e:
            print(f"预测失败 (样本 {i}): {str(e)}")
            continue
    
    # 计算评估指标
    evaluator = MedicalQAEvaluator()
    evaluation_result = evaluator.evaluate_batch(predictions, references)
    
    # 显示结果
    print("\n评估结果:")
    overall_metrics = evaluation_result['overall_metrics']
    for metric, value in overall_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 保存评估报告
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        report_path = os.path.join(args.output_dir, "evaluation_report.json")
        evaluator.save_evaluation_report(evaluation_result, report_path)
        print(f"\n评估报告已保存到: {report_path}")

def start_auto_learning(args):
    """启动自主学习"""
    print("启动自主学习...")
    
    # 创建自主学习配置
    config = AutoLearningConfig()
    if args.config:
        config_dict = load_training_config(args.config)
        auto_learning_config = config_dict.get('auto_learning', {})
        
        # 更新配置
        for key, value in auto_learning_config.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # 创建自主学习管理器
    auto_learning_manager = AutoLearningManager(config, args.data_dir)
    
    if args.force_retrain:
        success = auto_learning_manager.force_retrain("manual_force_retrain")
        if success:
            print("强制重训练已启动")
        else:
            print("无法启动强制重训练（可能已有训练进程在运行）")
    else:
        auto_learning_manager.start_autonomous_learning()
        print("自主学习已启动")
    
    # 显示学习状态
    status = auto_learning_manager.get_learning_status()
    print("\n学习状态:")
    for key, value in status.items():
        print(f"  {key}: {value}")

def show_status(args):
    """显示系统状态"""
    print("医疗问答系统状态:")
    
    # 检查数据目录
    data_dir = args.data_dir
    print(f"\n数据目录: {data_dir}")
    
    data_files = ['medical_data.json', 'knowledge_graph.json', 'feedback_data.json']
    for file_name in data_files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file_name} ({size} 字节)")
        else:
            print(f"  ✗ {file_name} (不存在)")
    
    # 检查模型目录
    model_dir = os.path.join("models", "medical_qa")
    print(f"\n模型目录: {model_dir}")
    
    if os.path.exists(model_dir):
        model_files = os.listdir(model_dir)
        print(f"  模型文件数量: {len(model_files)}")
        for file_name in model_files[:5]:  # 只显示前5个文件
            print(f"    - {file_name}")
        if len(model_files) > 5:
            print(f"    ... 还有 {len(model_files) - 5} 个文件")
    else:
        print("  ✗ 模型目录不存在")
    
    # 显示数据统计
    try:
        data_manager = TrainingDataManager(data_dir)
        stats = data_manager.get_data_statistics()
        print("\n数据统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"  获取数据统计失败: {str(e)}")

def create_config_template(args):
    """创建配置文件模板"""
    config_template = {
        "model_name": "hfl/chinese-bert-wwm-ext",
        "model_type": "seq2seq",
        "max_length": 512,
        "max_target_length": 128,
        "learning_rate": 2e-5,
        "batch_size": 4,
        "num_epochs": 3,
        "output_dir": "models/medical_qa",
        "auto_learning": {
            "min_feedback_count": 10,
            "min_low_score_ratio": 0.3,
            "feedback_score_threshold": 0.6,
            "retrain_interval_hours": 24,
            "incremental_learning": True,
            "max_training_samples": 1000
        }
    }
    
    config_path = args.output or "training_config.json"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_template, f, ensure_ascii=False, indent=2)
    
    print(f"配置文件模板已创建: {config_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="医疗问答模型训练脚本")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 通用参数
    parser.add_argument('--data-dir', default='api/data', help='数据目录路径')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    # 准备数据命令
    prepare_parser = subparsers.add_parser('prepare', help='准备训练数据')
    prepare_parser.add_argument('--output-dir', default='api/data', help='输出目录')
    prepare_parser.add_argument('--include-feedback', action='store_true', help='包含反馈数据')
    prepare_parser.add_argument('--train-ratio', type=float, default=0.8, help='训练集比例')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', default='training_config.json', help='训练配置文件')
    train_parser.add_argument('--output-dir', help='模型输出目录')
    train_parser.add_argument('--include-feedback', action='store_true', help='包含反馈数据')
    train_parser.add_argument('--train-ratio', type=float, default=0.8, help='训练集比例')
    train_parser.add_argument('--resume-from-checkpoint', help='从检查点恢复训练')
    train_parser.add_argument('--force', action='store_true', help='强制训练（即使数据不足）')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--model-path', default='models/medical_qa', help='模型路径')
    eval_parser.add_argument('--test-file', help='测试数据文件')
    eval_parser.add_argument('--output-dir', help='评估结果输出目录')
    
    # 自主学习命令
    auto_parser = subparsers.add_parser('auto-learn', help='启动自主学习')
    auto_parser.add_argument('--config', help='配置文件路径')
    auto_parser.add_argument('--force-retrain', action='store_true', help='强制重训练')
    
    # 状态命令
    status_parser = subparsers.add_parser('status', help='显示系统状态')
    
    # 配置模板命令
    config_parser = subparsers.add_parser('create-config', help='创建配置文件模板')
    config_parser.add_argument('--output', default='training_config.json', help='输出文件路径')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 执行命令
    if args.command == 'prepare':
        prepare_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'auto-learn':
        start_auto_learning(args)
    elif args.command == 'status':
        show_status(args)
    elif args.command == 'create-config':
        create_config_template(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()