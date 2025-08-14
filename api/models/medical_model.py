import json
import os
import logging
from collections import defaultdict

class MedicalQAModel:
    def __init__(self):
        self.diseases_data = {}
        self.departments_data = {}
        self.knowledge_graph = {}
        self._load_data()
        
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
    
    def generate_answer(self, question, context="", max_length=512):
        """
        根据问题和内置医疗知识生成答案
        Generate answer based on the question and internal medical knowledge
        """
        # Simple keyword-based matching for demonstration
        # In a real application, you would implement more sophisticated NLP techniques
        
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
                    answer = f"{symptom}可能与以下疾病相关：{diseases_str}。建议您咨询医生以获得准确诊断。"
                    return answer
        
        # Default response
        if "感冒" in question:
            return "感冒是一种常见的呼吸道疾病，通常由病毒引起。症状包括发热、咳嗽、流鼻涕等。建议多休息、多喝水，必要时可以服用感冒药物。如果症状严重或持续不改善，请及时就医。"
        
        if "胃炎" in question:
            return "胃炎是胃黏膜的炎症，常见症状包括胃痛、恶心、呕吐等。建议规律饮食，避免辛辣刺激食物，可适当服用胃黏膜保护剂。如果症状持续，请就医诊治。"
        
        return "感谢您的提问。我是基于医疗知识库的智能问答系统。根据您的问题，我无法提供具体的医疗建议。如果您有健康方面的担忧，建议咨询专业医生或前往医院就诊。"

# Global instance
medical_model = MedicalQAModel()