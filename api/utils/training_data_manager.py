"""
训练数据管理模块
Training Data Manager Module

负责处理医疗问答训练数据的预处理、格式化和管理
Handles preprocessing, formatting and management of medical QA training data
"""

import json
import os
import random
import logging
from typing import List, Dict, Tuple, Any
from datetime import datetime

class TrainingDataManager:
    """训练数据管理器"""
    
    def __init__(self, data_dir: str = None):
        """
        初始化训练数据管理器
        Initialize training data manager
        
        Args:
            data_dir: 数据目录路径 / Data directory path
        """
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        else:
            self.data_dir = data_dir
            
        self.medical_data = {}
        self.knowledge_graph = {}
        self.training_pairs = []
        self.feedback_data = []
        
        self._load_base_data()
        
    def _load_base_data(self):
        """加载基础医疗数据和知识图谱"""
        try:
            # 加载医疗数据
            medical_data_path = os.path.join(self.data_dir, 'medical_data.json')
            if os.path.exists(medical_data_path):
                with open(medical_data_path, 'r', encoding='utf-8') as f:
                    self.medical_data = json.load(f)
                    
            # 加载知识图谱
            kg_path = os.path.join(self.data_dir, 'knowledge_graph.json')
            if os.path.exists(kg_path):
                with open(kg_path, 'r', encoding='utf-8') as f:
                    self.knowledge_graph = json.load(f)
                    
            logging.info("基础医疗数据加载成功")
            
        except Exception as e:
            logging.error(f"加载基础数据失败: {str(e)}")
            raise
            
    def generate_qa_pairs_from_knowledge_base(self) -> List[Dict[str, str]]:
        """
        从知识库生成问答对
        Generate QA pairs from knowledge base
        
        Returns:
            List of QA pairs with question, answer, and context
        """
        qa_pairs = []
        
        # 从疾病数据生成问答对
        for disease in self.medical_data.get('diseases', []):
            disease_name = disease['name']
            description = disease['description']
            symptoms = disease.get('symptoms', [])
            drugs = disease.get('drugs', [])
            
            # 生成疾病描述问答
            qa_pairs.append({
                'question': f"什么是{disease_name}？",
                'answer': description,
                'context': f"疾病：{disease_name}",
                'category': 'disease_description'
            })
            
            qa_pairs.append({
                'question': f"{disease_name}是什么病？",
                'answer': description,
                'context': f"疾病：{disease_name}",
                'category': 'disease_description'
            })
            
            # 生成症状相关问答
            if symptoms:
                symptoms_str = '、'.join(symptoms)
                qa_pairs.append({
                    'question': f"{disease_name}有什么症状？",
                    'answer': f"{disease_name}的常见症状包括：{symptoms_str}。",
                    'context': f"疾病：{disease_name}，症状查询",
                    'category': 'symptoms'
                })
                
                qa_pairs.append({
                    'question': f"{disease_name}的症状是什么？",
                    'answer': f"患{disease_name}时可能出现以下症状：{symptoms_str}。",
                    'context': f"疾病：{disease_name}，症状查询",
                    'category': 'symptoms'
                })
            
            # 生成治疗药物问答
            if drugs:
                for drug in drugs:
                    drug_name = drug['name']
                    usage = drug['usage']
                    indication = drug.get('indication', '')
                    
                    qa_pairs.append({
                        'question': f"{disease_name}用什么药治疗？",
                        'answer': f"治疗{disease_name}可以使用{drug_name}，用法：{usage}。{indication}",
                        'context': f"疾病：{disease_name}，药物：{drug_name}",
                        'category': 'treatment'
                    })
                    
                    qa_pairs.append({
                        'question': f"{drug_name}怎么用？",
                        'answer': f"{drug_name}的用法是：{usage}。{indication}",
                        'context': f"药物：{drug_name}",
                        'category': 'drug_usage'
                    })
        
        # 从科室数据生成问答对
        for department in self.medical_data.get('departments', []):
            dept_name = department['name']
            symptoms = department.get('symptoms', [])
            
            if symptoms:
                symptoms_str = '、'.join(symptoms)
                qa_pairs.append({
                    'question': f"什么症状应该去{dept_name}？",
                    'answer': f"如果您有以下症状：{symptoms_str}，建议您前往{dept_name}就诊。",
                    'context': f"科室：{dept_name}",
                    'category': 'department_recommendation'
                })
                
                for symptom in symptoms:
                    qa_pairs.append({
                        'question': f"{symptom}应该挂什么科？",
                        'answer': f"如果您有{symptom}的症状，建议您可以考虑前往{dept_name}就诊。",
                        'context': f"症状：{symptom}，科室：{dept_name}",
                        'category': 'department_recommendation'
                    })
        
        # 从知识图谱生成问答对
        nodes = self.knowledge_graph.get('nodes', [])
        relationships = self.knowledge_graph.get('relationships', [])
        
        # 创建节点映射
        node_map = {node['id']: node for node in nodes}
        
        for rel in relationships:
            source_node = node_map.get(rel['source'])
            target_node = node_map.get(rel['target'])
            rel_type = rel['type']
            description = rel.get('description', '')
            
            if source_node and target_node:
                if rel_type == 'has_symptom':
                    qa_pairs.append({
                        'question': f"{source_node['name']}会有{target_node['name']}的症状吗？",
                        'answer': f"是的，{description}",
                        'context': f"疾病：{source_node['name']}，症状：{target_node['name']}",
                        'category': 'symptom_relation'
                    })
                    
                elif rel_type == 'treated_by':
                    qa_pairs.append({
                        'question': f"{source_node['name']}可以用{target_node['name']}治疗吗？",
                        'answer': f"是的，{description}",
                        'context': f"疾病：{source_node['name']}，药物：{target_node['name']}",
                        'category': 'treatment_relation'
                    })
                    
                elif rel_type == 'treated_in':
                    qa_pairs.append({
                        'question': f"{source_node['name']}应该去哪个科室？",
                        'answer': f"{description}",
                        'context': f"疾病：{source_node['name']}，科室：{target_node['name']}",
                        'category': 'department_relation'
                    })
        
        logging.info(f"从知识库生成了 {len(qa_pairs)} 个问答对")
        return qa_pairs
        
    def add_feedback_data(self, question: str, predicted_answer: str, 
                         correct_answer: str, score: float, feedback: str = ""):
        """
        添加用户反馈数据
        Add user feedback data
        
        Args:
            question: 用户问题
            predicted_answer: 模型预测答案
            correct_answer: 正确答案
            score: 用户评分 (0-1)
            feedback: 用户反馈文本
        """
        feedback_item = {
            'question': question,
            'predicted_answer': predicted_answer,
            'correct_answer': correct_answer,
            'score': score,
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        }
        
        self.feedback_data.append(feedback_item)
        
        # 自动保存反馈数据
        self._save_feedback_data()
        
    def _save_feedback_data(self):
        """保存反馈数据到文件"""
        feedback_path = os.path.join(self.data_dir, 'feedback_data.json')
        try:
            with open(feedback_path, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存反馈数据失败: {str(e)}")
            
    def load_feedback_data(self):
        """加载已有的反馈数据"""
        feedback_path = os.path.join(self.data_dir, 'feedback_data.json')
        if os.path.exists(feedback_path):
            try:
                with open(feedback_path, 'r', encoding='utf-8') as f:
                    self.feedback_data = json.load(f)
                logging.info(f"加载了 {len(self.feedback_data)} 条反馈数据")
            except Exception as e:
                logging.error(f"加载反馈数据失败: {str(e)}")
                self.feedback_data = []
                
    def prepare_training_data(self, include_feedback: bool = True, 
                            train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """
        准备训练数据
        Prepare training data
        
        Args:
            include_feedback: 是否包含反馈数据
            train_ratio: 训练集比例
            
        Returns:
            Tuple of (training_data, validation_data)
        """
        # 生成基础问答对
        qa_pairs = self.generate_qa_pairs_from_knowledge_base()
        
        # 包含反馈数据
        if include_feedback:
            self.load_feedback_data()
            for feedback in self.feedback_data:
                # 只包含高质量的反馈数据 (评分 > 0.7)
                if feedback['score'] > 0.7:
                    qa_pairs.append({
                        'question': feedback['question'],
                        'answer': feedback['correct_answer'],
                        'context': '用户反馈数据',
                        'category': 'feedback'
                    })
        
        # 随机打乱数据
        random.shuffle(qa_pairs)
        
        # 分割训练集和验证集
        split_idx = int(len(qa_pairs) * train_ratio)
        train_data = qa_pairs[:split_idx]
        val_data = qa_pairs[split_idx:]
        
        logging.info(f"准备训练数据完成：训练集 {len(train_data)} 条，验证集 {len(val_data)} 条")
        
        return train_data, val_data
        
    def save_training_data(self, train_data: List[Dict], val_data: List[Dict], 
                          output_dir: str = None):
        """
        保存训练数据到文件
        Save training data to files
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            output_dir: 输出目录
        """
        if output_dir is None:
            output_dir = self.data_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存训练数据
        train_path = os.path.join(output_dir, 'train_data.json')
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
            
        # 保存验证数据
        val_path = os.path.join(output_dir, 'val_data.json')
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
            
        logging.info(f"训练数据已保存到 {output_dir}")
        
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息
        Get data statistics
        
        Returns:
            Dictionary containing data statistics
        """
        qa_pairs = self.generate_qa_pairs_from_knowledge_base()
        
        # 按类别统计
        category_counts = {}
        for pair in qa_pairs:
            category = pair.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
            
        stats = {
            'total_qa_pairs': len(qa_pairs),
            'category_distribution': category_counts,
            'total_diseases': len(self.medical_data.get('diseases', [])),
            'total_departments': len(self.medical_data.get('departments', [])),
            'total_feedback': len(self.feedback_data),
            'knowledge_graph_nodes': len(self.knowledge_graph.get('nodes', [])),
            'knowledge_graph_relationships': len(self.knowledge_graph.get('relationships', []))
        }
        
        return stats