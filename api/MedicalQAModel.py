import json
import jieba
import re
from collections import defaultdict
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class :
    def __init__(self, knowledge_graph_path, medical_data_path):
        self.knowledge_graph_path = knowledge_graph_path
        self.medical_data_path = medical_data_path
        self.diseases_data = []
        self.departments_data = []
        self.qa_pairs = []
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = None
        self.questions = []
        self.answers = []
        
    def load_data(self):
        """加载医疗数据"""
        # 加载知识图谱数据
        with open(self.knowledge_graph_path, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
            
        # 加载医疗数据
        with open(self.medical_data_path, 'r', encoding='utf-8') as f:
            medical_data = json.load(f)
            
        self.diseases_data = medical_data['diseases']
        self.departments_data = medical_data['departments']
        
        # 构建问答对
        self._build_qa_pairs()
        
    def _build_qa_pairs(self):
        """根据医疗数据构建问答对"""
        # 疾病相关问答
        for disease in self.diseases_data:
            name = disease['name']
            description = disease['description']
            symptoms = disease['symptoms']
            drugs = disease['drugs']
            
            # 疾病描述问答
            self.qa_pairs.append({
                'question': f'什么是{name}？',
                'answer': description,
                'type': 'disease_description'
            })
            
            self.qa_pairs.append({
                'question': f'{name}的症状有哪些？',
                'answer': '、'.join(symptoms),
                'type': 'disease_symptoms'
            })
            
            # 疾病用药问答
            drug_info = '；'.join([f"{drug['name']}({drug['usage']})" for drug in drugs])
            self.qa_pairs.append({
                'question': f'{name}怎么治疗？',
                'answer': drug_info,
                'type': 'disease_treatment'
            })
            
            self.qa_pairs.append({
                'question': f'{name}用什么药？',
                'answer': drug_info,
                'type': 'disease_drugs'
            })
            
            # 症状相关问答
            for symptom in symptoms:
                self.qa_pairs.append({
                    'question': f'{symptom}可能是什么病？',
                    'answer': name,
                    'type': 'symptom_disease'
                })
                
            # 科室相关问答
            for dept in self.departments_data:
                if any(symptom in dept['symptoms'] for symptom in symptoms):
                    self.qa_pairs.append({
                        'question': f'{name}应该挂哪个科室？',
                        'answer': dept['name'],
                        'type': 'disease_department'
                    })
                    break
        
        # 科室相关问答
        for dept in self.departments_data:
            name = dept['name']
            symptoms = dept['symptoms']
            
            self.qa_pairs.append({
                'question': f'{name}主要治疗什么症状？',
                'answer': '、'.join(symptoms),
                'type': 'department_symptoms'
            })
            
            for symptom in symptoms:
                self.qa_pairs.append({
                    'question': f'{symptom}应该挂哪个科室？',
                    'answer': name,
                    'type': 'symptom_department'
                })
                
        print(f"构建了 {len(self.qa_pairs)} 个问答对")
        
    def preprocess_text(self, text):
        """文本预处理"""
        # 去除标点符号
        text = re.sub(r'[^\w\s]', '', text)
        # 分词
        words = jieba.lcut(text)
        # 去除停用词（简单处理）
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        words = [w for w in words if w not in stop_words and len(w) > 1]
        return ' '.join(words)
    
    def train(self):
        """训练模型"""
        # 准备训练数据
        self.questions = [self.preprocess_text(qa['question']) for qa in self.qa_pairs]
        self.answers = [qa['answer'] for qa in self.qa_pairs]
        
        # 训练TF-IDF向量器
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
        print("模型训练完成")
        
    def predict(self, question, top_k=3):
        """预测答案"""
        # 预处理问题
        processed_question = self.preprocess_text(question)
        
        # 向量化
        question_vector = self.vectorizer.transform([processed_question])
        
        # 计算相似度
        similarities = cosine_similarity(question_vector, self.question_vectors).flatten()
        
        # 获取最相似的问题
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # 设置相似度阈值
                results.append({
                    'question': self.qa_pairs[idx]['question'],
                    'answer': self.answers[idx],
                    'similarity': similarities[idx]
                })
                
        return results
    
    def evaluate(self, test_questions=None):
        """评估模型"""
        if test_questions is None:
            # 使用部分训练数据作为测试数据
            test_questions = random.sample(self.qa_pairs, min(10, len(self.qa_pairs)))
            
        correct = 0
        total = len(test_questions)
        
        for qa in test_questions:
            results = self.predict(qa['question'], top_k=1)
            if results and results[0]['question'] == qa['question']:
                correct += 1
                
        accuracy = correct / total if total > 0 else 0
        print(f"模型准确率: {accuracy:.2%} ({correct}/{total})")
        return accuracy

def main():
    # 初始化模型
    model = MedicalQAModel(
        knowledge_graph_path='data/knowledge_graph.json',
        medical_data_path='data/medical_data.json'
    )
    
    # 加载数据
    model.load_data()
    
    # 训练模型
    model.train()
    
    # 评估模型
    model.evaluate()
    
    # 测试一些问题
    test_questions = [
        "感冒的症状有哪些？",
        "胃痛应该挂哪个科室？",
        "什么是冠心病？",
        "奥美拉唑怎么用？"
    ]
    
    print("\n=== 问答测试 ===")
    for question in test_questions:
        print(f"\n问题: {question}")
        results = model.predict(question)
        if results:
            for i, result in enumerate(results[:2]):  # 显示前两个结果
                print(f"  相似问题{i+1}: {result['question']}")
                print(f"  答案: {result['answer']}")
                print(f"  相似度: {result['similarity']:.2f}")
        else:
            print("  未找到相关答案")

if __name__ == "__main__":
    main()