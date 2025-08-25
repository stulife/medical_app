"""
模型评估和验证模块
Model Evaluation and Validation Module

提供模型性能评估、指标计算和验证功能
Provides model performance evaluation, metrics calculation and validation functionality
"""

import json
import os
import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import re
from collections import defaultdict

# NLP评估指标
try:
    from rouge_score import rouge_scorer
    from sacrebleu import sentence_bleu, BLEU
    ROUGE_AVAILABLE = True
    BLEU_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    BLEU_AVAILABLE = False
    logging.warning("ROUGE和BLEU评估库未安装，将使用基础评估指标")

from difflib import SequenceMatcher
import jieba

class MedicalQAEvaluator:
    """医疗问答模型评估器"""
    
    def __init__(self, use_jieba: bool = True):
        """
        初始化评估器
        
        Args:
            use_jieba: 是否使用jieba分词
        """
        self.use_jieba = use_jieba
        if use_jieba:
            try:
                import jieba
                self.jieba = jieba
            except ImportError:
                logging.warning("jieba未安装，将使用字符级别评估")
                self.use_jieba = False
                
        # 初始化ROUGE评分器
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        
    def calculate_exact_match(self, predicted: str, reference: str) -> float:
        """
        计算完全匹配得分
        
        Args:
            predicted: 预测答案
            reference: 参考答案
            
        Returns:
            完全匹配得分 (0 或 1)
        """
        # 标准化文本
        pred_normalized = self._normalize_text(predicted)
        ref_normalized = self._normalize_text(reference)
        
        return 1.0 if pred_normalized == ref_normalized else 0.0
    
    def calculate_f1_score(self, predicted: str, reference: str) -> float:
        """
        计算F1得分
        
        Args:
            predicted: 预测答案
            reference: 参考答案
            
        Returns:
            F1得分
        """
        pred_tokens = self._tokenize(predicted)
        ref_tokens = self._tokenize(reference)
        
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
            
        # 计算交集
        common_tokens = set(pred_tokens) & set(ref_tokens)
        
        if len(common_tokens) == 0:
            return 0.0
            
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(ref_tokens)
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def calculate_bleu_score(self, predicted: str, reference: str) -> float:
        """
        计算BLEU得分
        
        Args:
            predicted: 预测答案
            reference: 参考答案
            
        Returns:
            BLEU得分
        """
        if not BLEU_AVAILABLE:
            # 使用简化的n-gram重叠计算
            return self._simple_bleu(predicted, reference)
            
        pred_tokens = self._tokenize(predicted)
        ref_tokens = self._tokenize(reference)
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
            
        try:
            # 使用sacrebleu计算BLEU
            bleu_score = sentence_bleu(pred_tokens, [ref_tokens])
            return bleu_score.score / 100.0  # 转换为0-1范围
        except Exception:
            return self._simple_bleu(predicted, reference)
    
    def calculate_rouge_score(self, predicted: str, reference: str) -> Dict[str, float]:
        """
        计算ROUGE得分
        
        Args:
            predicted: 预测答案
            reference: 参考答案
            
        Returns:
            ROUGE得分字典
        """
        if not ROUGE_AVAILABLE:
            # 使用简化的重叠计算
            return self._simple_rouge(predicted, reference)
            
        try:
            scores = self.rouge_scorer.score(reference, predicted)
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception:
            return self._simple_rouge(predicted, reference)
    
    def calculate_semantic_similarity(self, predicted: str, reference: str) -> float:
        """
        计算语义相似度
        
        Args:
            predicted: 预测答案
            reference: 参考答案
            
        Returns:
            语义相似度得分
        """
        # 使用SequenceMatcher计算序列相似度
        similarity = SequenceMatcher(None, predicted, reference).ratio()
        return similarity
    
    def calculate_medical_accuracy(self, predicted: str, reference: str, 
                                 medical_entities: List[str] = None) -> float:
        """
        计算医疗准确性
        
        Args:
            predicted: 预测答案
            reference: 参考答案
            medical_entities: 医疗实体列表
            
        Returns:
            医疗准确性得分
        """
        if medical_entities is None:
            medical_entities = self._extract_medical_entities(reference)
            
        pred_entities = self._extract_medical_entities(predicted)
        
        if len(medical_entities) == 0:
            return 1.0 if len(pred_entities) == 0 else 0.5
            
        # 计算医疗实体的匹配度
        matched_entities = 0
        for entity in medical_entities:
            if any(entity in pred_entity or pred_entity in entity for pred_entity in pred_entities):
                matched_entities += 1
                
        accuracy = matched_entities / len(medical_entities)
        return accuracy
    
    def evaluate_single_prediction(self, predicted: str, reference: str, 
                                 context: str = "", category: str = "") -> Dict[str, float]:
        """
        评估单个预测结果
        
        Args:
            predicted: 预测答案
            reference: 参考答案
            context: 上下文
            category: 问题类别
            
        Returns:
            评估指标字典
        """
        metrics = {}
        
        # 基础指标
        metrics['exact_match'] = self.calculate_exact_match(predicted, reference)
        metrics['f1_score'] = self.calculate_f1_score(predicted, reference)
        metrics['semantic_similarity'] = self.calculate_semantic_similarity(predicted, reference)
        
        # BLEU得分
        metrics['bleu_score'] = self.calculate_bleu_score(predicted, reference)
        
        # ROUGE得分
        rouge_scores = self.calculate_rouge_score(predicted, reference)
        metrics.update(rouge_scores)
        
        # 医疗准确性
        metrics['medical_accuracy'] = self.calculate_medical_accuracy(predicted, reference)
        
        # 长度比较
        metrics['length_ratio'] = len(predicted) / max(len(reference), 1)
        
        # 答案完整性 (是否包含关键信息)
        metrics['completeness'] = self._calculate_completeness(predicted, reference)
        
        return metrics
    
    def evaluate_batch(self, predictions: List[str], references: List[str],
                      contexts: List[str] = None, categories: List[str] = None) -> Dict[str, Any]:
        """
        批量评估预测结果
        
        Args:
            predictions: 预测答案列表
            references: 参考答案列表
            contexts: 上下文列表
            categories: 问题类别列表
            
        Returns:
            批量评估结果
        """
        if len(predictions) != len(references):
            raise ValueError("预测和参考答案数量不匹配")
            
        if contexts is None:
            contexts = [""] * len(predictions)
        if categories is None:
            categories = [""] * len(predictions)
            
        all_metrics = []
        category_metrics = defaultdict(list)
        
        for i, (pred, ref, ctx, cat) in enumerate(zip(predictions, references, contexts, categories)):
            metrics = self.evaluate_single_prediction(pred, ref, ctx, cat)
            all_metrics.append(metrics)
            
            if cat:
                category_metrics[cat].append(metrics)
        
        # 计算总体统计
        overall_metrics = self._aggregate_metrics(all_metrics)
        
        # 计算分类别统计
        category_stats = {}
        for category, metrics_list in category_metrics.items():
            category_stats[category] = self._aggregate_metrics(metrics_list)
        
        return {
            'overall_metrics': overall_metrics,
            'category_metrics': category_stats,
            'individual_metrics': all_metrics,
            'total_samples': len(predictions)
        }
    
    def _normalize_text(self, text: str) -> str:
        """标准化文本"""
        # 移除多余空格和标点
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[，。！？；：""''（）【】]', '', text)
        return text.lower()
    
    def _tokenize(self, text: str) -> List[str]:
        """文本分词"""
        if self.use_jieba and hasattr(self, 'jieba'):
            return list(self.jieba.cut(text))
        else:
            # 字符级别分词
            return list(text)
    
    def _simple_bleu(self, predicted: str, reference: str) -> float:
        """简化的BLEU计算"""
        pred_tokens = self._tokenize(predicted)
        ref_tokens = self._tokenize(reference)
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
            
        # 计算1-gram和2-gram重叠
        pred_1grams = set(pred_tokens)
        ref_1grams = set(ref_tokens)
        
        overlap_1 = len(pred_1grams & ref_1grams)
        precision_1 = overlap_1 / len(pred_1grams) if pred_1grams else 0
        
        # 简化计算，只考虑1-gram
        return precision_1
    
    def _simple_rouge(self, predicted: str, reference: str) -> Dict[str, float]:
        """简化的ROUGE计算"""
        pred_tokens = self._tokenize(predicted)
        ref_tokens = self._tokenize(reference)
        
        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            return {'rouge1': 1.0, 'rouge2': 1.0, 'rougeL': 1.0}
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
            
        # ROUGE-1 (unigram重叠)
        pred_1grams = set(pred_tokens)
        ref_1grams = set(ref_tokens)
        overlap_1 = len(pred_1grams & ref_1grams)
        
        precision_1 = overlap_1 / len(pred_1grams)
        recall_1 = overlap_1 / len(ref_1grams)
        rouge1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
        
        # ROUGE-L (最长公共子序列)
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)
        precision_l = lcs_length / len(pred_tokens)
        recall_l = lcs_length / len(ref_tokens)
        rougeL = 2 * precision_l * recall_l / (precision_l + recall_l) if (precision_l + recall_l) > 0 else 0
        
        return {
            'rouge1': rouge1,
            'rouge2': 0.0,  # 简化实现不计算2-gram
            'rougeL': rougeL
        }
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """计算最长公共子序列长度"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                    
        return dp[m][n]
    
    def _extract_medical_entities(self, text: str) -> List[str]:
        """提取医疗实体"""
        # 简化实现：基于关键词匹配
        medical_keywords = [
            '症状', '疾病', '药物', '治疗', '诊断', '科室', '医院', '医生',
            '发热', '咳嗽', '头痛', '胃痛', '腹痛', '恶心', '呕吐', '腹泻',
            '感冒', '肺炎', '胃炎', '高血压', '糖尿病', '哮喘', '冠心病',
            '阿司匹林', '青霉素', '布洛芬', '奥美拉唑', '二甲双胍'
        ]
        
        entities = []
        for keyword in medical_keywords:
            if keyword in text:
                entities.append(keyword)
                
        return entities
    
    def _calculate_completeness(self, predicted: str, reference: str) -> float:
        """计算答案完整性"""
        # 基于关键信息覆盖度
        ref_keywords = self._extract_medical_entities(reference)
        pred_keywords = self._extract_medical_entities(predicted)
        
        if len(ref_keywords) == 0:
            return 1.0
            
        covered_keywords = 0
        for keyword in ref_keywords:
            if keyword in predicted:
                covered_keywords += 1
                
        return covered_keywords / len(ref_keywords)
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """聚合指标"""
        if not metrics_list:
            return {}
            
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
                aggregated[f"{key}_min"] = np.min(values)
                aggregated[f"{key}_max"] = np.max(values)
                
        return aggregated
    
    def save_evaluation_report(self, evaluation_result: Dict[str, Any], 
                             output_path: str):
        """
        保存评估报告
        
        Args:
            evaluation_result: 评估结果
            output_path: 输出路径
        """
        # 添加时间戳和元信息
        report = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_result': evaluation_result,
            'metrics_description': {
                'exact_match': '完全匹配率',
                'f1_score': 'F1得分',
                'bleu_score': 'BLEU得分',
                'rouge1': 'ROUGE-1得分',
                'rougeL': 'ROUGE-L得分',
                'semantic_similarity': '语义相似度',
                'medical_accuracy': '医疗准确性',
                'completeness': '答案完整性'
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        logging.info(f"评估报告已保存到: {output_path}")

def create_evaluator() -> MedicalQAEvaluator:
    """创建默认评估器"""
    return MedicalQAEvaluator(use_jieba=True)