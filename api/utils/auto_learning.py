"""
自主学习和模型更新机制
Autonomous Learning and Model Update Mechanism

实现基于用户反馈的自主学习和模型持续改进功能
Implements autonomous learning and continuous model improvement based on user feedback
"""

import os
import json
import logging
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from .training_data_manager import TrainingDataManager
from .model_evaluator import MedicalQAEvaluator
from ..models.qa_trainer import MedicalQATrainer, ModelConfig

class AutoLearningConfig:
    """自主学习配置"""
    
    def __init__(self):
        # 学习触发条件
        self.min_feedback_count = 10  # 最小反馈数量
        self.min_low_score_ratio = 0.3  # 低分反馈比例阈值
        self.feedback_score_threshold = 0.6  # 反馈评分阈值
        
        # 训练配置
        self.retrain_interval_hours = 24  # 重训练间隔（小时）
        self.incremental_learning = True  # 是否增量学习
        self.max_training_samples = 1000  # 最大训练样本数
        
        # 模型更新策略
        self.backup_model_count = 3  # 保留的模型备份数量
        self.performance_improvement_threshold = 0.05  # 性能提升阈值
        
        # 质量控制
        self.min_answer_length = 5  # 最小答案长度
        self.max_answer_length = 500  # 最大答案长度
        self.duplicate_threshold = 0.9  # 重复内容阈值

class AutoLearningManager:
    """自主学习管理器"""
    
    def __init__(self, config: AutoLearningConfig = None, data_dir: str = None):
        """
        初始化自主学习管理器
        
        Args:
            config: 自主学习配置
            data_dir: 数据目录
        """
        self.config = config or AutoLearningConfig()
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), '..', 'data')
        
        # 初始化组件
        self.data_manager = TrainingDataManager(self.data_dir)
        self.evaluator = MedicalQAEvaluator()
        self.current_trainer = None
        
        # 学习状态
        self.is_learning = False
        self.last_training_time = None
        self.learning_thread = None
        self.learning_stats = {
            'total_retrains': 0,
            'successful_updates': 0,
            'performance_improvements': 0,
            'last_performance_score': 0.0
        }
        
        # 加载历史状态
        self._load_learning_state()
        
    def _load_learning_state(self):
        """加载学习状态"""
        state_path = os.path.join(self.data_dir, 'learning_state.json')
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    self.learning_stats = state.get('stats', self.learning_stats)
                    if state.get('last_training_time'):
                        self.last_training_time = datetime.fromisoformat(state['last_training_time'])
                logging.info("学习状态加载成功")
            except Exception as e:
                logging.error(f"加载学习状态失败: {str(e)}")
    
    def _save_learning_state(self):
        """保存学习状态"""
        state = {
            'stats': self.learning_stats,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'timestamp': datetime.now().isoformat()
        }
        
        state_path = os.path.join(self.data_dir, 'learning_state.json')
        try:
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存学习状态失败: {str(e)}")
    
    def add_feedback(self, question: str, predicted_answer: str, 
                    correct_answer: str, score: float, feedback: str = ""):
        """
        添加用户反馈
        
        Args:
            question: 用户问题
            predicted_answer: 模型预测答案
            correct_answer: 正确答案
            score: 用户评分 (0-1)
            feedback: 用户反馈文本
        """
        # 质量检查
        if not self._validate_feedback(question, correct_answer, score):
            logging.warning("反馈数据质量检查未通过，已忽略")
            return
            
        # 添加到数据管理器
        self.data_manager.add_feedback_data(
            question, predicted_answer, correct_answer, score, feedback
        )
        
        logging.info(f"添加反馈数据：问题='{question[:50]}...', 评分={score}")
        
        # 检查是否需要触发学习
        self._check_learning_trigger()
    
    def _validate_feedback(self, question: str, answer: str, score: float) -> bool:
        """验证反馈数据质量"""
        # 检查基本格式
        if not question or not answer:
            return False
            
        if not (0 <= score <= 1):
            return False
            
        # 检查答案长度
        if len(answer) < self.config.min_answer_length or len(answer) > self.config.max_answer_length:
            return False
            
        # 检查是否重复（简化实现）
        existing_feedback = self.data_manager.feedback_data
        for feedback in existing_feedback[-10:]:  # 只检查最近10条
            if (feedback['question'] == question and 
                self._text_similarity(feedback['correct_answer'], answer) > self.config.duplicate_threshold):
                return False
                
        return True
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _check_learning_trigger(self):
        """检查是否需要触发学习"""
        if self.is_learning:
            return
            
        # 加载反馈数据
        self.data_manager.load_feedback_data()
        feedback_data = self.data_manager.feedback_data
        
        if len(feedback_data) < self.config.min_feedback_count:
            return
            
        # 检查时间间隔
        if (self.last_training_time and 
            datetime.now() - self.last_training_time < timedelta(hours=self.config.retrain_interval_hours)):
            return
            
        # 分析反馈质量
        recent_feedback = feedback_data[-50:]  # 分析最近50条反馈
        low_score_count = sum(1 for f in recent_feedback if f['score'] < self.config.feedback_score_threshold)
        low_score_ratio = low_score_count / len(recent_feedback)
        
        # 判断是否需要重训练
        if low_score_ratio >= self.config.min_low_score_ratio:
            logging.info(f"检测到模型性能下降（低分比例：{low_score_ratio:.2f}），触发自主学习")
            self.start_autonomous_learning()
        else:
            logging.info(f"模型性能良好（低分比例：{low_score_ratio:.2f}），暂不需要重训练")
    
    def start_autonomous_learning(self):
        """启动自主学习"""
        if self.is_learning:
            logging.warning("自主学习已在进行中")
            return
            
        self.is_learning = True
        
        # 在后台线程中进行学习
        self.learning_thread = threading.Thread(target=self._autonomous_learning_worker)
        self.learning_thread.daemon = True
        self.learning_thread.start()
        
        logging.info("自主学习已启动")
    
    def _autonomous_learning_worker(self):
        """自主学习工作线程"""
        try:
            logging.info("开始自主学习过程...")
            
            # 1. 准备训练数据
            train_data, val_data = self._prepare_learning_data()
            
            if len(train_data) < 10:
                logging.warning("训练数据不足，停止自主学习")
                return
                
            # 2. 备份当前模型
            self._backup_current_model()
            
            # 3. 训练新模型
            new_model_path = self._train_improved_model(train_data, val_data)
            
            if new_model_path is None:
                logging.error("模型训练失败")
                return
                
            # 4. 评估新模型
            improvement = self._evaluate_model_improvement(new_model_path, val_data)
            
            # 5. 决定是否更新模型
            if improvement > self.config.performance_improvement_threshold:
                self._update_production_model(new_model_path)
                self.learning_stats['successful_updates'] += 1
                self.learning_stats['performance_improvements'] += 1
                logging.info(f"模型更新成功，性能提升：{improvement:.3f}")
            else:
                logging.info(f"新模型性能提升不足（{improvement:.3f}），保持当前模型")
                
            # 6. 更新统计信息
            self.learning_stats['total_retrains'] += 1
            self.last_training_time = datetime.now()
            self._save_learning_state()
            
        except Exception as e:
            logging.error(f"自主学习过程出错: {str(e)}")
        finally:
            self.is_learning = False
            logging.info("自主学习过程结束")
    
    def _prepare_learning_data(self) -> Tuple[List[Dict], List[Dict]]:
        """准备学习数据"""
        # 获取基础训练数据
        base_train_data, base_val_data = self.data_manager.prepare_training_data(
            include_feedback=True, train_ratio=0.8
        )
        
        # 限制训练样本数量
        if len(base_train_data) > self.config.max_training_samples:
            # 优先保留高质量的反馈数据
            feedback_data = [d for d in base_train_data if d.get('category') == 'feedback']
            other_data = [d for d in base_train_data if d.get('category') != 'feedback']
            
            remaining_slots = self.config.max_training_samples - len(feedback_data)
            if remaining_slots > 0:
                base_train_data = feedback_data + other_data[:remaining_slots]
            else:
                base_train_data = feedback_data[:self.config.max_training_samples]
        
        logging.info(f"准备了 {len(base_train_data)} 个训练样本，{len(base_val_data)} 个验证样本")
        return base_train_data, base_val_data
    
    def _backup_current_model(self):
        """备份当前模型"""
        # 实现模型备份逻辑
        models_dir = os.path.join(self.data_dir, '..', 'models')
        current_model_dir = os.path.join(models_dir, 'medical_qa')
        
        if not os.path.exists(current_model_dir):
            return
            
        # 创建备份目录
        backup_dir = os.path.join(models_dir, 'backups')
        os.makedirs(backup_dir, exist_ok=True)
        
        # 生成备份名称
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"medical_qa_backup_{timestamp}")
        
        # 复制模型文件
        import shutil
        try:
            shutil.copytree(current_model_dir, backup_path)
            logging.info(f"模型备份成功：{backup_path}")
            
            # 清理旧备份
            self._cleanup_old_backups(backup_dir)
            
        except Exception as e:
            logging.error(f"模型备份失败：{str(e)}")
    
    def _cleanup_old_backups(self, backup_dir: str):
        """清理旧备份"""
        try:
            backups = [d for d in os.listdir(backup_dir) if d.startswith('medical_qa_backup_')]
            backups.sort(reverse=True)  # 按时间倒序
            
            # 保留最新的几个备份
            for old_backup in backups[self.config.backup_model_count:]:
                old_backup_path = os.path.join(backup_dir, old_backup)
                import shutil
                shutil.rmtree(old_backup_path)
                logging.info(f"清理旧备份：{old_backup}")
                
        except Exception as e:
            logging.error(f"清理备份失败：{str(e)}")
    
    def _train_improved_model(self, train_data: List[Dict], val_data: List[Dict]) -> Optional[str]:
        """训练改进的模型"""
        try:
            # 配置训练参数
            config = ModelConfig(
                model_name="hfl/chinese-bert-wwm-ext",
                model_type="seq2seq",
                max_length=512,
                max_target_length=128,
                learning_rate=1e-5,  # 较小的学习率用于微调
                batch_size=4,
                num_epochs=2,  # 较少的epoch避免过拟合
                output_dir=os.path.join(self.data_dir, '..', 'models', 'medical_qa_improved')
            )
            
            # 创建训练器
            trainer = MedicalQATrainer(config)
            
            # 如果存在当前模型，从其继续训练
            current_model_path = os.path.join(self.data_dir, '..', 'models', 'medical_qa')
            if os.path.exists(current_model_path):
                try:
                    trainer.load_trained_model(current_model_path)
                    logging.info("从当前模型继续训练")
                except Exception as e:
                    logging.warning(f"加载当前模型失败，从预训练模型开始：{str(e)}")
                    trainer.load_model_and_tokenizer()
            else:
                trainer.load_model_and_tokenizer()
            
            # 开始训练
            train_result = trainer.train(train_data, val_data)
            
            # 保存训练信息
            self._save_training_log(train_result, config.output_dir)
            
            return config.output_dir
            
        except Exception as e:
            logging.error(f"模型训练失败：{str(e)}")
            return None
    
    def _save_training_log(self, train_result: Dict[str, Any], model_dir: str):
        """保存训练日志"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'train_result': train_result,
            'learning_stats': self.learning_stats,
            'trigger_reason': 'autonomous_learning'
        }
        
        log_path = os.path.join(model_dir, 'training_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    def _evaluate_model_improvement(self, new_model_path: str, val_data: List[Dict]) -> float:
        """评估模型改进程度"""
        try:
            # 加载新模型
            new_trainer = MedicalQATrainer()
            new_trainer.load_trained_model(new_model_path)
            
            # 生成预测结果
            predictions = []
            references = []
            
            for item in val_data[:50]:  # 评估前50个样本
                question = item['question']
                context = item.get('context', '')
                reference = item['answer']
                
                try:
                    prediction = new_trainer.predict(question, context)
                    predictions.append(prediction)
                    references.append(reference)
                except Exception as e:
                    logging.warning(f"预测失败：{str(e)}")
                    continue
            
            if not predictions:
                return 0.0
                
            # 计算评估指标
            evaluation_result = self.evaluator.evaluate_batch(predictions, references)
            new_score = evaluation_result['overall_metrics'].get('f1_score_mean', 0.0)
            
            # 计算改进程度
            improvement = new_score - self.learning_stats['last_performance_score']
            self.learning_stats['last_performance_score'] = new_score
            
            logging.info(f"模型评估完成：新模型F1={new_score:.3f}, 改进={improvement:.3f}")
            return improvement
            
        except Exception as e:
            logging.error(f"模型评估失败：{str(e)}")
            return 0.0
    
    def _update_production_model(self, new_model_path: str):
        """更新生产模型"""
        production_model_path = os.path.join(self.data_dir, '..', 'models', 'medical_qa')
        
        try:
            # 移除旧的生产模型
            if os.path.exists(production_model_path):
                import shutil
                shutil.rmtree(production_model_path)
            
            # 复制新模型到生产路径
            import shutil
            shutil.copytree(new_model_path, production_model_path)
            
            logging.info(f"生产模型更新成功：{production_model_path}")
            
        except Exception as e:
            logging.error(f"更新生产模型失败：{str(e)}")
            raise
    
    def get_learning_status(self) -> Dict[str, Any]:
        """获取学习状态"""
        return {
            'is_learning': self.is_learning,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'learning_stats': self.learning_stats,
            'feedback_count': len(self.data_manager.feedback_data) if hasattr(self.data_manager, 'feedback_data') else 0,
            'config': {
                'min_feedback_count': self.config.min_feedback_count,
                'retrain_interval_hours': self.config.retrain_interval_hours,
                'min_low_score_ratio': self.config.min_low_score_ratio
            }
        }
    
    def force_retrain(self, reason: str = "manual_trigger"):
        """强制重新训练"""
        if self.is_learning:
            logging.warning("已有学习进程在运行，无法强制重训练")
            return False
            
        logging.info(f"强制重训练触发，原因：{reason}")
        self.start_autonomous_learning()
        return True

# 全局自主学习管理器实例
auto_learning_manager = None

def get_auto_learning_manager(data_dir: str = None) -> AutoLearningManager:
    """获取自主学习管理器实例"""
    global auto_learning_manager
    if auto_learning_manager is None:
        auto_learning_manager = AutoLearningManager(data_dir=data_dir)
    return auto_learning_manager