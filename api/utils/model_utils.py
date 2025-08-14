import torch
from typing import Dict, List, Any
import logging

def format_qa_prompt(question: str, context: str = "") -> str:
    """
    格式化问答提示
    Format QA prompt
    """
    if context:
        return f"背景：{context}\n问题：{question}\n答案："
    else:
        return f"问题：{question}\n答案："

def calculate_model_score(response: str, expected_keywords: List[str] = None) -> float:
    """
    计算模型回答的评分
    Calculate model response score
    """
    if not response or len(response.strip()) == 0:
        return 0.0
    
    # 基础长度评分
    length_score = min(len(response) / 100.0, 1.0)
    
    # 关键词匹配评分
    keyword_score = 0.0
    if expected_keywords:
        matched_keywords = sum(1 for keyword in expected_keywords if keyword in response)
        keyword_score = matched_keywords / len(expected_keywords) if expected_keywords else 0
    
    # 综合评分
    final_score = 0.5 * length_score + 0.5 * (keyword_score if expected_keywords else length_score)
    
    return round(final_score, 2)

def prepare_fine_tuning_data(feedback_data: List[Dict[str, Any]]) -> Dict[str, List]:
    """
    准备微调数据
    Prepare fine-tuning data
    """
    train_data = {
        "questions": [],
        "answers": [],
        "scores": []
    }
    
    for item in feedback_data:
        train_data["questions"].append(item["question"])
        train_data["answers"].append(item["answer"])
        train_data["scores"].append(item["score"])
    
    return train_data

class ConversationContext:
    """
    管理多轮对话上下文
    Manage multi-turn conversation context
    """
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history
    
    def add_turn(self, question: str, answer: str):
        """
        添加对话轮次
        Add conversation turn
        """
        self.history.append({
            "question": question,
            "answer": answer
        })
        
        # 保持历史记录在限制范围内
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_context(self) -> str:
        """
        获取对话上下文
        Get conversation context
        """
        context_parts = []
        for turn in self.history:
            context_parts.append(f"问：{turn['question']} 答：{turn['answer']}")
        
        return " ".join(context_parts)
    
    def clear(self):
        """
        清除对话历史
        Clear conversation history
        """
        self.history = []