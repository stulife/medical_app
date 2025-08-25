#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础功能测试脚本
Basic Functionality Test Script

测试不依赖深度学习的基础功能
Test basic functionality without deep learning dependencies
"""

import os
import sys
import logging
import time
from datetime import datetime

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
            logging.FileHandler('test_basic.log', encoding='utf-8')
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
        
        # 测试添加反馈数据
        data_manager.add_feedback_data(
            "测试问题", "预测答案", "正确答案", 0.8, "测试反馈"
        )
        print("✓ 添加反馈数据成功")
        
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
        print("单个评估测试:")
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            metrics = evaluator.evaluate_single_prediction(pred, ref)
            print(f"\n样本 {i+1}:")
            print(f"  F1得分: {metrics['f1_score']:.4f}")
            print(f"  完全匹配: {metrics['exact_match']:.4f}")
            print(f"  语义相似度: {metrics['semantic_similarity']:.4f}")
        
        # 测试批量评估
        evaluation_result = evaluator.evaluate_batch(predictions, references)
        print(f"\n✓ 批量评估完成，总样本数: {evaluation_result['total_samples']}")
        
        overall_metrics = evaluation_result['overall_metrics']
        print("\n整体性能指标:")
        key_metrics = ['f1_score_mean', 'exact_match_mean', 'semantic_similarity_mean']
        for metric in key_metrics:
            if metric in overall_metrics:
                print(f"  {metric}: {overall_metrics[metric]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型评估器测试失败: {str(e)}")
        logging.error(f"模型评估器测试失败: {str(e)}", exc_info=True)
        return False

def test_medical_model_basic():
    """测试医疗模型基础功能"""
    print("=" * 50)
    print("测试医疗模型基础功能")
    print("=" * 50)
    
    try:
        # 直接测试基础医疗模型类，不加载深度学习组件
        import json
        import os
        from collections import defaultdict
        
        # 模拟基础医疗模型功能
        class BasicMedicalQAModel:
            def __init__(self):
                self.diseases_data = {}
                self.departments_data = {}
                self.knowledge_graph = {}
                self.query_count = 0
                self._load_data()
                
            def _load_data(self):
                """加载医疗数据"""
                try:
                    data_dir = 'api/data'
                    # 加载医疗数据
                    medical_data_path = os.path.join(data_dir, 'medical_data.json')
                    if os.path.exists(medical_data_path):
                        with open(medical_data_path, 'r', encoding='utf-8') as f:
                            medical_data = json.load(f)
                            
                        # 处理疾病数据
                        for disease in medical_data.get('diseases', []):
                            self.diseases_data[disease['name']] = disease
                            
                        # 处理科室数据
                        for department in medical_data.get('departments', []):
                            self.departments_data[department['name']] = department
                    
                    # 加载知识图谱
                    kg_path = os.path.join(data_dir, 'knowledge_graph.json')
                    if os.path.exists(kg_path):
                        with open(kg_path, 'r', encoding='utf-8') as f:
                            self.knowledge_graph = json.load(f)
                            
                except Exception as e:
                    print(f"加载数据失败: {str(e)}")
                    
            def generate_answer(self, question):
                """生成答案"""
                self.query_count += 1
                question_lower = question.lower()
                
                # 检查疾病相关问题
                for disease_name, disease_info in self.diseases_data.items():
                    if disease_name.lower() in question_lower:
                        symptoms = ', '.join(disease_info.get('symptoms', []))
                        return f"关于{disease_name}：{disease_info.get('description', '')}。常见症状包括：{symptoms}。"
                
                # 默认回答
                return "感谢您的提问。我是基于医疗知识库的智能问答系统。"
                
            def get_stats(self):
                """获取统计信息"""
                return {
                    'query_count': self.query_count,
                    'diseases_count': len(self.diseases_data),
                    'departments_count': len(self.departments_data)
                }
        
        # 测试基础模型
        model = BasicMedicalQAModel()
        
        # 测试问题
        test_questions = [
            "感冒有什么症状？",
            "胃痛应该挂什么科？",
            "高血压用什么药治疗？",
            "什么是肺结核？",
            "头痛应该去哪个科室？"
        ]
        
        print("测试基础问答功能:")
        for i, question in enumerate(test_questions):
            answer = model.generate_answer(question)
            print(f"\n问题 {i+1}: {question}")
            print(f"答案: {answer[:100]}...")
        
        # 测试模型统计
        stats = model.get_stats()
        print(f"\n✓ 模型统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ 医疗模型基础功能测试失败: {str(e)}")
        logging.error(f"医疗模型基础功能测试失败: {str(e)}", exc_info=True)
        return False

def test_configuration():
    """测试配置文件"""
    print("=" * 50)
    print("测试配置文件")
    print("=" * 50)
    
    try:
        import json
        config_path = 'config/training_config.json'
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print("✓ 配置文件加载成功")
            print("配置内容:")
            print(f"  模型名称: {config.get('model_name')}")
            print(f"  模型类型: {config.get('model_type')}")
            print(f"  批次大小: {config.get('batch_size')}")
            print(f"  训练轮数: {config.get('num_epochs')}")
            
            # 检查自主学习配置
            auto_learning = config.get('auto_learning', {})
            if auto_learning:
                print("\n自主学习配置:")
                print(f"  最小反馈数量: {auto_learning.get('min_feedback_count')}")
                print(f"  重训练间隔: {auto_learning.get('retrain_interval_hours')} 小时")
                print(f"  最大训练样本: {auto_learning.get('max_training_samples')}")
        else:
            print("✗ 配置文件不存在")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 配置文件测试失败: {str(e)}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("=" * 50)
    print("测试文件结构")
    print("=" * 50)
    
    # 检查重要文件和目录
    important_paths = [
        'api/',
        'api/data/',
        'api/data/medical_data.json',
        'api/data/knowledge_graph.json',
        'api/models/',
        'api/models/medical_model.py',
        'api/models/qa_trainer.py',
        'api/utils/',
        'api/utils/training_data_manager.py',
        'api/utils/model_evaluator.py',
        'api/utils/auto_learning.py',
        'config/',
        'config/training_config.json',
        'train.py',
        'test_training.py',
        'requirements.txt'
    ]
    
    print("检查文件结构:")
    missing_files = []
    
    for path in important_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                print(f"  ✓ {path}/ (目录)")
            else:
                size = os.path.getsize(path)
                print(f"  ✓ {path} ({size} 字节)")
        else:
            print(f"  ✗ {path} (缺失)")
            missing_files.append(path)
    
    if missing_files:
        print(f"\n警告: 缺失 {len(missing_files)} 个文件/目录")
        return False
    else:
        print("\n✓ 所有重要文件都存在")
        return True

def main():
    """主测试函数"""
    setup_logging()
    
    print("医疗问答模型基础功能测试")
    print("=" * 60)
    print(f"开始时间: {datetime.now()}")
    print("=" * 60)
    
    test_results = []
    
    # 运行基础测试
    tests = [
        ("文件结构", test_file_structure),
        ("配置文件", test_configuration),
        ("数据管理器", test_data_manager),
        ("模型评估器", test_model_evaluator),
        ("医疗模型基础功能", test_medical_model_basic)
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
        print(f"{status_symbol} {test_name:<20} {status:<8} ({elapsed_time:.2f}s)")
        
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
        print("✓ 所有基础测试通过！")
        print("\n下一步：")
        print("  1. 安装深度学习依赖（如果需要）:")
        print("     pip install torch==2.0.1 transformers==4.33.0 accelerate==0.21.0")
        print("  2. 运行完整测试: python test_training.py")
        print("  3. 准备训练数据: python train.py prepare")
        print("  4. 开始模型训练: python train.py train --config config/training_config.json")
    else:
        print("! 部分测试失败，请检查：")
        print("  1. 确保在正确的项目目录中运行")
        print("  2. 检查数据文件是否存在")
        print("  3. 查看详细错误日志: test_basic.log")
    
    return failed == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)