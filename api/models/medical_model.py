import json
import os
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any
from datetime import datetime

# 导入深度学习相关模块
try:
    from .qa_trainer import MedicalQATrainer
    from ..utils.auto_learning import get_auto_learning_manager
    from ..utils.model_evaluator import MedicalQAEvaluator
    DEEP_LEARNING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"深度学习模块导入失败: {str(e)}，将使用基础问答功能")
    DEEP_LEARNING_AVAILABLE = False

class MedicalQAModel:
    def __init__(self, use_deep_learning: bool = True):
        self.diseases_data = {}
        self.departments_data = {}
        self.knowledge_graph = {}
        self.use_deep_learning = use_deep_learning and DEEP_LEARNING_AVAILABLE
        
        # 深度学习模型相关
        self.dl_trainer = None
        self.auto_learning_manager = None
        self.evaluator = None
        self.model_loaded = False
        
        # 统计信息
        self.query_count = 0
        self.feedback_count = 0
        
        self._load_data()
        
        # 初始化深度学习组件
        if self.use_deep_learning:
            self._initialize_deep_learning_components()
        
    def _load_data(self):
        """
        加载医疗数据
        Load medical data from JSON files
        """
        try:
            # Load medical data
            medical_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'medical_data.json')
            if os.path.exists(medical_data_path):
                with open(medical_data_path, 'r', encoding='utf-8') as f:
                    medical_data = json.load(f)
                    
                # Process diseases data
                for disease in medical_data.get('diseases', []):
                    self.diseases_data[disease['name']] = disease
                    
                # Process departments data
                for department in medical_data.get('departments', []):
                    self.departments_data[department['name']] = department
            
            # Load knowledge graph
            kg_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge_graph.json')
            if os.path.exists(kg_path):
                with open(kg_path, 'r', encoding='utf-8') as f:
                    self.knowledge_graph = json.load(f)
                    
            logging.info("Medical data loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load medical data: {str(e)}")
            raise
    
    def _initialize_deep_learning_components(self):
        """初始化深度学习组件"""
        try:
            # 初始化评估器
            self.evaluator = MedicalQAEvaluator()
            
            # 初始化自主学习管理器
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            self.auto_learning_manager = get_auto_learning_manager(data_dir)
            
            # 尝试加载已训练的模型
            self._load_trained_model()
            
            logging.info("深度学习组件初始化成功")
            
        except Exception as e:
            logging.error(f"深度学习组件初始化失败: {str(e)}")
            self.use_deep_learning = False
    
    def _load_trained_model(self):
        """加载已训练的深度学习模型"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'medical_qa')
            
            if os.path.exists(model_path):
                self.dl_trainer = MedicalQATrainer()
                self.dl_trainer.load_trained_model(model_path)
                self.model_loaded = True
                logging.info("已训练的深度学习模型加载成功")
            else:
                logging.info("未找到已训练的模型，将使用基础问答功能")
                
        except Exception as e:
            logging.error(f"加载深度学习模型失败: {str(e)}")
            self.model_loaded = False
    
    def generate_answer(self, question: str, context: str = "", max_length: int = 512, 
                       session_id: str = "", user_feedback: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        根据问题和内置医疗知识生成答案
        Generate answer based on the question and internal medical knowledge
        
        Args:
            question: 用户问题
            context: 上下文信息
            max_length: 最大答案长度
            session_id: 会话ID
            user_feedback: 用户反馈信息
            
        Returns:
            包含答案、置信度、来源等信息的字典
        """
        self.query_count += 1
        
        # 处理用户反馈
        if user_feedback and self.auto_learning_manager:
            self._process_user_feedback(user_feedback)
        
        # 尝试使用深度学习模型
        if self.use_deep_learning and self.model_loaded:
            try:
                dl_answer = self._generate_deep_learning_answer(question, context)
                if dl_answer and self._validate_answer(dl_answer):
                    return {
                        'question': question,
                        'answer': dl_answer,
                        'source': 'deep_learning',
                        'score': 0.85,
                        'session_id': session_id,
                        'timestamp': datetime.now().isoformat()
                    }
            except Exception as e:
                logging.warning(f"深度学习推理失败，回退到基础方法: {str(e)}")
        
        # 使用基础知识库方法
        basic_answer = self._generate_basic_answer(question, context)
        
        return {
            'question': question,
            'answer': basic_answer,
            'source': 'knowledge_base',
            'score': 0.75,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_deep_learning_answer(self, question: str, context: str) -> str:
        """使用深度学习模型生成答案"""
        if not self.dl_trainer:
            raise ValueError("深度学习模型未加载")
            
        answer = self.dl_trainer.predict(question, context)
        return answer.strip()
    
    def _generate_basic_answer(self, question: str, context: str) -> str:
        """使用基础知识库生成答案"""
        question_lower = question.lower()
        
        # Check for disease-related questions
        for disease_name, disease_info in self.diseases_data.items():
            if disease_name.lower() in question_lower:
                symptoms = ', '.join(disease_info.get('symptoms', []))
                drugs = disease_info.get('drugs', [])
                drug_info = '; '.join([f"{drug['name']} ({drug['usage']})" for drug in drugs])
                
                answer = f"关于{disease_name}：{disease_info.get('description', '')}。"
                answer += f"常见症状包括：{symptoms}。"
                if drugs:
                    answer += f"常用药物有：{drug_info}。"
                
                # 添加科室推荐
                related_departments = self._find_related_departments(disease_info.get('symptoms', []))
                if related_departments:
                    depts_str = '、'.join(related_departments)
                    answer += f"建议就诊科室：{depts_str}。"
                
                return answer
        
        # Check for department-related questions
        for dept_name, dept_info in self.departments_data.items():
            if dept_name.lower() in question_lower or any(symptom.lower() in question_lower for symptom in dept_info.get('symptoms', [])):
                symptoms = ', '.join(dept_info.get('symptoms', []))
                answer = f"如果您有以下症状：{symptoms}，建议您前往{dept_name}就诊。"
                return answer
                
        # Check for symptom-related questions
        all_symptoms = set()
        for disease_info in self.diseases_data.values():
            all_symptoms.update(disease_info.get('symptoms', []))
            
        for symptom in all_symptoms:
            if symptom.lower() in question_lower:
                related_diseases = []
                for disease_name, disease_info in self.diseases_data.items():
                    if symptom in disease_info.get('symptoms', []):
                        related_diseases.append(disease_name)
                
                if related_diseases:
                    diseases_str = '、'.join(related_diseases)
                    answer = f"{symptom}可能与以下疾病相关：{diseases_str}。"
                    
                    # 添加科室推荐
                    related_departments = self._find_related_departments([symptom])
                    if related_departments:
                        depts_str = '、'.join(related_departments)
                        answer += f"建议就诊科室：{depts_str}。"
                    
                    answer += "建议您咨询医生以获得准确诊断。"
                    return answer
        
        # 推荐科室功能 - 当用户直接询问应该去哪个科室时
        if any(keyword in question_lower for keyword in ["挂哪个科", "看哪个科", "去哪个科", "推荐科室", "哪个科室", "应该挂号"]):
            answer = self._recommend_departments(question)
            if answer:
                return answer
        
        # Default response
        if "感冒" in question:
            return "感冒是一种常见的呼吸道疾病，通常由病毒引起。症状包括发热、咳嗽、流鼻涕等。建议多休息、多喝水，必要时可以服用感冒药物。如果症状严重或持续不改善，请及时就医。"
        
        if "胃炎" in question:
            return "胃炎是胃黏膜的炎症，常见症状包括胃痛、恶心、呕吐等。建议规律饮食，避免辛辣刺激食物，可适当服用胃黏膜保护剂。如果症状持续，请就医诊治。"
        if "你好" in question or "您好" in question:
            return "我是一个智能助手，我可以回答你的问题。请输入你的问题。"
        if "谢谢" in question:
            return "不客气。"
        if "再见" in question:
            return "祝你生活愉快"
        return "感谢您的提问。我是基于医疗知识库的智能问答系统。根据您的问题，我无法提供具体的医疗建议。如果您有健康方面的担忧，建议咨询专业医生或前往医院就诊。"
    
    def _validate_answer(self, answer: str) -> bool:
        """验证答案质量"""
        if not answer or len(answer.strip()) < 5:
            return False
            
        # 检查是否包含有害内容（简化实现）
        harmful_keywords = ['自杀', '死亡', '危险', '毒性']
        for keyword in harmful_keywords:
            if keyword in answer:
                return False
                
        return True
    
    def _process_user_feedback(self, feedback: Dict[str, Any]):
        """处理用户反馈"""
        try:
            question = feedback.get('question', '')
            predicted_answer = feedback.get('predicted_answer', '')
            correct_answer = feedback.get('correct_answer', '')
            score = feedback.get('score', 0.0)
            feedback_text = feedback.get('feedback', '')
            
            if self.auto_learning_manager:
                self.auto_learning_manager.add_feedback(
                    question, predicted_answer, correct_answer, score, feedback_text
                )
                self.feedback_count += 1
                
        except Exception as e:
            logging.error(f"处理用户反馈失败: {str(e)}")
    
    def add_feedback(self, question: str, predicted_answer: str, 
                    correct_answer: str, score: float, feedback: str = ""):
        """添加用户反馈（向后兼容）"""
        feedback_data = {
            'question': question,
            'predicted_answer': predicted_answer,
            'correct_answer': correct_answer,
            'score': score,
            'feedback': feedback
        }
        self._process_user_feedback(feedback_data)
    
    def get_model_stats(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        stats = {
            'query_count': self.query_count,
            'feedback_count': self.feedback_count,
            'use_deep_learning': self.use_deep_learning,
            'model_loaded': self.model_loaded,
            'diseases_count': len(self.diseases_data),
            'departments_count': len(self.departments_data)
        }
        
        # 添加自主学习状态
        if self.auto_learning_manager:
            learning_status = self.auto_learning_manager.get_learning_status()
            stats['learning_status'] = learning_status
            
        return stats
    
    def train_model(self, force_retrain: bool = False) -> bool:
        """训练或更新模型"""
        if not self.use_deep_learning:
            logging.warning("深度学习功能未启用")
            return False
            
        if not self.auto_learning_manager:
            logging.error("自主学习管理器未初始化")
            return False
            
        if force_retrain:
            return self.auto_learning_manager.force_retrain("manual_request")
        else:
            # 检查是否满足自动训练条件
            learning_status = self.auto_learning_manager.get_learning_status()
            if not learning_status['is_learning']:
                self.auto_learning_manager.start_autonomous_learning()
                return True
            else:
                logging.info("模型正在训练中")
                return False
    
    def evaluate_model(self, test_questions: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """评估模型性能"""
        if not self.use_deep_learning or not self.evaluator:
            return {'error': '深度学习功能未启用'}
            
        # 使用默认测试问题或用户提供的问题
        if test_questions is None:
            test_questions = self._get_default_test_questions()
            
        predictions = []
        references = []
        
        for item in test_questions:
            question = item['question']
            expected_answer = item['answer']
            context = item.get('context', '')
            
            # 生成预测
            result = self.generate_answer(question, context)
            predicted_answer = result['answer']
            
            predictions.append(predicted_answer)
            references.append(expected_answer)
            
        # 计算评估指标
        if predictions and references:
            evaluation_result = self.evaluator.evaluate_batch(predictions, references)
            return evaluation_result
        else:
            return {'error': '没有有效的测试数据'}
    
    def _get_default_test_questions(self) -> List[Dict[str, str]]:
        """获取默认测试问题"""
        return [
            {
                'question': '感冒有什么症状？',
                'answer': '感冒的常见症状包括：发热、咳嗽、流鼻涕、头痛、乏力、咽喉痛。',
                'context': '疾病症状查询'
            },
            {
                'question': '胃痛应该挂什么科？',
                'answer': '如果您有胃痛的症状，建议您可以考虑前往消化内科就诊。',
                'context': '科室推荐'
            },
            {
                'question': '高血压用什么药？',
                'answer': '治疗高血压可以使用硝苯地平，用法：一次10mg，一日3次。扩张血管，降低血压',
                'context': '药物治疗'
            }
        ]

    def _find_related_departments(self, symptoms):
        """
        根据症状列表查找相关科室
        Find related departments based on symptoms list
        """
        related_departments = []
        
        for symptom in symptoms:
            for dept_name, dept_info in self.departments_data.items():
                if symptom in dept_info.get('symptoms', []):
                    # 避免重复添加科室
                    if dept_name not in related_departments:
                        related_departments.append(dept_name)
        
        return related_departments

    def _recommend_departments(self, question):
        """
        根据问题内容推荐科室
        Recommend departments based on question content
        """
        question_lower = question.lower()
        matched_departments = []
        
        # 根据科室症状关键词匹配
        for dept_name, dept_info in self.departments_data.items():
            symptoms = dept_info.get('symptoms', [])
            # 如果科室有50%以上的症状关键词在问题中出现，则推荐该科室
            matched_symptoms = [symptom for symptom in symptoms if symptom.lower() in question_lower]
            if len(matched_symptoms) >= len(symptoms) * 0.5 or len(matched_symptoms) >= 3:
                matched_departments.append(dept_name)
        
        if matched_departments:
            depts_str = '、'.join(matched_departments)
            return f"根据您的症状描述，建议您可以考虑挂以下科室：{depts_str}。"
        
        # 如果没有明确匹配，提供常见科室推荐
        common_depts = ["内科", "外科", "急诊科"]
        available_common_depts = [dept for dept in common_depts if dept in self.departments_data]
        if available_common_depts:
            depts_str = '、'.join(available_common_depts)
            return f"如果不确定具体科室，您可以先考虑挂以下常见科室：{depts_str}，医生会根据具体情况为您安排。"
        
        return None

# Global instance
medical_model = MedicalQAModel()