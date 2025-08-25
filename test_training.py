#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医疗问答模型训练和推理测试脚本
Medical QA Model Training and Inference Test Script

测试模型训练、推理和自主学习功能
Test model training, inference and autonomous learning functionality
"""

import os
import sys
import logging
import time
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_training.log', encoding='utf-8')
        ]
    )

def test_data_manager():
    """测试数据管理器"""
    print("=" * 50)
    print("测试训练数据管理器")
    print("=" * 50)
    
    try:
        from api.utils.training_data_manager import TrainingDataManager
        
        # 创建数据管理器
        data_manager = TrainingDataManager('api/data')
        
        # 测试生成问答对
        qa_pairs = data_manager.generate_qa_pairs_from_knowledge_base()
        print(f"✓ 生成问答对: {len(qa_pairs)} 个")
        
        # 显示几个示例
        print("\n问答对示例:")
        for i, pair in enumerate(qa_pairs[:3]):
            print(f"  {i+1}. 问题: {pair['question']}")
            print(f"     答案: {pair['answer'][:50]}...")
            print(f"     类别: {pair['category']}")
            print()
        
        # 测试数据统计
        stats = data_manager.get_data_statistics()
        print("数据统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 测试准备训练数据
        train_data, val_data = data_manager.prepare_training_data()
        print(f"\n✓ 准备训练数据: 训练集 {len(train_data)} 个, 验证集 {len(val_data)} 个")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据管理器测试失败: {str(e)}")
        logging.error(f"数据管理器测试失败: {str(e)}", exc_info=True)
        return False

def test_model_evaluator():
    """测试模型评估器"""
    print("=" * 50)
    print("测试模型评估器")
    print("=" * 50)
    
    try:
        from api.utils.model_evaluator import MedicalQAEvaluator
        
        # 创建评估器
        evaluator = MedicalQAEvaluator()
        
        # 测试数据
        predictions = [
            "感冒的常见症状包括发热、咳嗽、流鼻涕、头痛等。",
            "胃痛建议前往消化内科就诊。",
            "高血压可以使用硝苯地平治疗。"
        ]
        
        references = [
            "感冒是由病毒引起的上呼吸道感染，症状包括发热、咳嗽、流鼻涕。",
            "如果您有胃痛症状，建议前往消化内科就诊。",
            "治疗高血压可以使用硝苯地平，用法：一次10mg，一日3次。"
        ]
        
        # 测试单个评估
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            metrics = evaluator.evaluate_single_prediction(pred, ref)
            print(f"\n样本 {i+1} 评估结果:")
            for metric, score in metrics.items():
                print(f"  {metric}: {score:.4f}")
        
        # 测试批量评估
        evaluation_result = evaluator.evaluate_batch(predictions, references)
        print(f"\n✓ 批量评估完成，总样本数: {evaluation_result['total_samples']}")
        
        overall_metrics = evaluation_result['overall_metrics']
        print("\n整体性能指标:")
        for metric, value in overall_metrics.items():
            if 'mean' in metric:
                print(f"  {metric}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型评估器测试失败: {str(e)}")
        logging.error(f"模型评估器测试失败: {str(e)}", exc_info=True)
        return False

def test_medical_model():
    """测试医疗模型"""
    print("=" * 50)
    print("测试医疗模型")
    print("=" * 50)
    
    try:
        from api.models.medical_model import MedicalQAModel
        
        # 创建模型（不使用深度学习以避免依赖问题）
        model = MedicalQAModel(use_deep_learning=True)
        
        # 测试问题
        test_questions = [
            "胃痛应该挂什么科？",
            "高血压用什么药治疗？",
            "什么是肺结核？",
            "头痛应该去哪个科室？",
            "脑梗塞用什么药治疗？",
            "什么叫心绞痛？",
            "感冒有哪些症状？"
        ]
        
        print("测试基础问答功能:")
        for i, question in enumerate(test_questions):
            # result = model.generate_answer(question, session_id=f"test_session_{i}")
            result = model._generate_deep_learning_answer(question,context=f"")
            print(f"\n问题 {i+1}: {question}")
            # 修改: 处理result可能为字符串的情况
            if isinstance(result, dict):
                print(f"答案: {result['answer']}")
                print(f"来源: {result['source']}")
                print(f"置信度: {result['confidence']}")
            else:
                print(f"{str(result)}")
        
        # 测试模型统计
        stats = model.get_model_stats()
        print(f"\n✓ 模型统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ 医疗模型测试失败: {str(e)}")
        logging.error(f"医疗模型测试失败: {str(e)}", exc_info=True)
        return False

def test_model_trainer():
    """测试模型训练器（轻量级测试）"""
    print("=" * 50)
    print("测试模型训练器")
    print("=" * 50)
    
    try:
        from api.models.qa_trainer import ModelConfig, MedicalQATrainer
        from api.utils.training_data_manager import TrainingDataManager
        
        # 创建小规模测试配置
        config = ModelConfig(
            model_name="hfl/chinese-bert-wwm-ext",
            model_type="seq2seq",
            max_length=128,
            max_target_length=64,
            learning_rate=5e-5,
            batch_size=2,
            num_epochs=1,  # 只训练1个epoch
            output_dir="test_models/medical_qa_test"
        )
        
        print(f"✓ 创建训练配置: {config.model_name}")
        
        # 准备小规模训练数据
        data_manager = TrainingDataManager('api/data')
        train_data, val_data = data_manager.prepare_training_data(train_ratio=0.8)
        
        # 只使用少量数据进行测试
        test_train_data = train_data[:5]  # 只使用5个训练样本
        test_val_data = val_data[:2]      # 只使用2个验证样本
        
        print(f"✓ 准备测试数据: 训练 {len(test_train_data)} 个, 验证 {len(test_val_data)} 个")
        
        # 由于可能没有GPU或者依赖不完整，我们只测试组件创建
        trainer = MedicalQATrainer(config)
        print("✓ 创建训练器成功")
        
        # 测试数据集创建
        try:
            train_dataset, val_dataset = trainer.prepare_datasets(test_train_data, test_val_data)
            print(f"✓ 创建数据集: 训练集 {len(train_dataset)} 个, 验证集 {len(val_dataset)} 个")
        except Exception as e:
            print(f"! 数据集创建失败（可能缺少模型文件）: {str(e)}")
        
        print("✓ 模型训练器基础功能测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 模型训练器测试失败: {str(e)}")
        logging.error(f"模型训练器测试失败: {str(e)}", exc_info=True)
        return False

def test_auto_learning():
    """测试自主学习功能"""
    print("=" * 50)
    print("测试自主学习功能")
    print("=" * 50)
    
    try:
        from api.utils.auto_learning import AutoLearningManager, AutoLearningConfig
        
        # 创建测试配置
        config = AutoLearningConfig()
        config.min_feedback_count = 3  # 降低测试阈值
        config.retrain_interval_hours = 0.1  # 降低时间间隔
        
        # 创建自主学习管理器
        manager = AutoLearningManager(config, 'api/data')
        print("✓ 创建自主学习管理器成功")
        
        # 测试添加反馈
        test_feedback = [
            {
                'question': '感冒症状是什么？',
                'predicted_answer': '感冒有发热症状',
                'correct_answer': '感冒的症状包括发热、咳嗽、流鼻涕、头痛、乏力等',
                'score': 0.5,
                'feedback': '答案不够完整'
            },
            {
                'question': '胃痛挂什么科？',
                'predicted_answer': '内科',
                'correct_answer': '消化内科',
                'score': 0.4,
                'feedback': '科室不准确'
            },
            {
                'question': '高血压用药？',
                'predicted_answer': '降压药',
                'correct_answer': '硝苯地平，一次10mg，一日3次',
                'score': 0.3,
                'feedback': '需要具体药名和用法'
            }
        ]
        
        for feedback in test_feedback:
            manager.add_feedback(
                feedback['question'],
                feedback['predicted_answer'],
                feedback['correct_answer'],
                feedback['score'],
                feedback['feedback']
            )
        
        print(f"✓ 添加 {len(test_feedback)} 条反馈数据")
        
        # 获取学习状态
        status = manager.get_learning_status()
        print("\n学习状态:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ 自主学习功能测试失败: {str(e)}")
        logging.error(f"自主学习功能测试失败: {str(e)}", exc_info=True)
        return False

def test_integration():
    """集成测试"""
    print("=" * 50)
    print("集成测试")
    print("=" * 50)
    
    try:
        from api.models.medical_model import MedicalQAModel
        
        # 创建模型
        model = MedicalQAModel(use_deep_learning=False)
        
        # 模拟用户交互流程
        questions = [
            "我发烧了怎么办？",
            "胃痛严重吗？",
            "什么药治疗感冒？"
        ]
        
        session_id = "integration_test_session"
        
        for i, question in enumerate(questions):
            print(f"\n--- 第 {i+1} 轮对话 ---")
            print(f"用户问题: {question}")
            
            # 生成答案
            result = model.generate_answer(question, session_id=session_id)
            answer = result['answer']
            # print(f"系统回答: {str(result)}...")
            print(f"系统回答: {answer}...")
            
            # 模拟用户反馈
            feedback_score = 0.6 + i * 0.1  # 逐渐提高评分
            corrected_answer = f"关于{question}的详细解答..."
            
            feedback_data = {
                'question': question,
                'predicted_answer': answer,
                'correct_answer': corrected_answer,
                'score': feedback_score,
                'feedback': f'第{i+1}次反馈'
            }
            
            # 添加反馈
            model._process_user_feedback(feedback_data)
            print(f"用户反馈: 评分 {feedback_score}")
        
        # 显示最终统计
        final_stats = model.get_model_stats()
        print(f"\n✓ 集成测试完成")
        print(f"  总查询次数: {final_stats['query_count']}")
        print(f"  总反馈次数: {final_stats['feedback_count']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 集成测试失败: {str(e)}")
        logging.error(f"集成测试失败: {str(e)}", exc_info=True)
        return False

def main():
    """主测试函数"""
    setup_logging()
    
    print("医疗问答模型训练和推理系统测试")
    print("=" * 60)
    print(f"开始时间: {datetime.now()}")
    print("=" * 60)
    
    test_results = []
    
    # 运行所有测试
    tests = [
        ("数据管理器", test_data_manager),
        ("模型评估器", test_model_evaluator),
        ("医疗模型", test_medical_model),
        ("模型训练器", test_model_trainer),
        ("自主学习", test_auto_learning),
        ("集成测试", test_integration)
    ]
    
    for test_name, test_func in tests:
        print(f"\n开始测试: {test_name}")
        start_time = time.time()
        
        try:
            success = test_func()
            elapsed_time = time.time() - start_time
            
            if success:
                print(f"✓ {test_name} 测试通过 ({elapsed_time:.2f}s)")
                test_results.append((test_name, "通过", elapsed_time))
            else:
                print(f"✗ {test_name} 测试失败 ({elapsed_time:.2f}s)")
                test_results.append((test_name, "失败", elapsed_time))
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"✗ {test_name} 测试出错: {str(e)} ({elapsed_time:.2f}s)")
            test_results.append((test_name, "出错", elapsed_time))
    
    # 显示测试总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, status, elapsed_time in test_results:
        status_symbol = "✓" if status == "通过" else "✗"
        print(f"{status_symbol} {test_name:<15} {status:<8} ({elapsed_time:.2f}s)")
        
        if status == "通过":
            passed += 1
        else:
            failed += 1
    
    print("-" * 60)
    print(f"总计: {len(test_results)} 个测试")
    print(f"通过: {passed} 个")
    print(f"失败: {failed} 个")
    print(f"成功率: {passed/len(test_results)*100:.1f}%")
    print(f"结束时间: {datetime.now()}")
    
    # 提供使用建议
    print("\n" + "=" * 60)
    print("使用建议")
    print("=" * 60)
    
    if failed == 0:
        print("✓ 所有测试通过！可以开始使用训练功能：")
        print("  1. 运行 'python train.py prepare' 准备训练数据")
        print("  2. 运行 'python train.py train' 开始训练模型")
        print("  3. 运行 'python train.py evaluate' 评估模型性能")
        print("  4. 运行 'python train.py auto-learn' 启动自主学习")
    else:
        print("! 部分测试失败，请检查：")
        print("  1. 确保已安装所有依赖: pip install -r requirements.txt")
        print("  2. 检查数据文件是否存在于 api/data/ 目录")
        print("  3. 查看详细错误日志: test_training.log")
    
    return failed == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)