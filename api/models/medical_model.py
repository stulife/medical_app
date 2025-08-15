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