"""
医疗问答模型训练器
Medical QA Model Trainer

基于Transformers实现的医疗问答模型训练和微调功能
Medical QA model training and fine-tuning based on Transformers
"""

import os
import json
import torch
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    pipeline
)
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

@dataclass
class ModelConfig:
    """模型配置类"""
    model_name: str = "bert-base-chinese"  # 默认使用中文BERT
    model_type: str = "seq2seq"  # "qa" for extractive QA, "seq2seq" for generative QA
    max_length: int = 512
    max_target_length: int = 128
    learning_rate: float = 2e-5
    batch_size: int = 8
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_steps: int = 500
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    output_dir: str = "models/medical_qa"
    logging_steps: int = 100

class MedicalQADataset(Dataset):
    """医疗问答数据集类"""
    
    def __init__(self, data: List[Dict], tokenizer, config: ModelConfig):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        context = item.get('context', '')
        
        if self.config.model_type == "seq2seq":
            # 对于生成式模型
            input_text = f"问题: {question}"
            if context:
                input_text = f"背景: {context} 问题: {question}"
                
            # 编码输入
            model_inputs = self.tokenizer(
                input_text,
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # 编码目标
            labels = self.tokenizer(
                answer,
                max_length=self.config.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            model_inputs["labels"] = labels["input_ids"]
            
            # 将张量降维
            for key in model_inputs:
                model_inputs[key] = model_inputs[key].squeeze()
                
            return model_inputs
            
        else:
            # 对于抽取式问答模型 (暂不实现)
            raise NotImplementedError("Extractive QA model not implemented yet")

class MedicalQATrainer:
    """医疗问答模型训练器"""
    
    def __init__(self, config: ModelConfig = None):
        """
        初始化训练器
        
        Args:
            config: 模型配置
        """
        self.config = config or ModelConfig()
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {self.device}")
        
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        try:
            if self.config.model_type == "seq2seq":
                # 使用T5或BART等生成式模型
                if "t5" in self.config.model_name.lower():
                    self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
                else:
                    # 尝试使用其他生成式模型
                    self.tokenizer = AutoTokenizer.from_pretrained("ClueAI/ChatYuan-large-v2")
                    self.model = AutoModelForSeq2SeqLM.from_pretrained("ClueAI/ChatYuan-large-v2")
            else:
                # 抽取式问答模型
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self.model = AutoModelForQuestionAnswering.from_pretrained(self.config.model_name)
                
            # 确保tokenizer有pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model.to(self.device)
            logging.info(f"成功加载模型: {self.config.model_name}")
            
        except Exception as e:
            logging.error(f"加载模型失败: {str(e)}")
            # 使用备用模型
            logging.info("尝试使用备用模型...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
                self.model = AutoModelForSeq2SeqLM.from_pretrained("hfl/chinese-bert-wwm-ext", 
                                                                 force_download=False,
                                                                 local_files_only=False)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.to(self.device)
                logging.info("备用模型加载成功")
            except Exception as e2:
                logging.error(f"备用模型加载也失败: {str(e2)}")
                raise
    
    def prepare_datasets(self, train_data: List[Dict], val_data: List[Dict]) -> Tuple[Dataset, Dataset]:
        """
        准备训练和验证数据集
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            
        Returns:
            训练数据集和验证数据集
        """
        train_dataset = MedicalQADataset(train_data, self.tokenizer, self.config)
        val_dataset = MedicalQADataset(val_data, self.tokenizer, self.config)
        
        logging.info(f"训练数据集大小: {len(train_dataset)}")
        logging.info(f"验证数据集大小: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        
        # 对于生成式模型，需要解码预测结果
        if self.config.model_type == "seq2seq":
            # 这里简化处理，实际应该解码并计算BLEU、ROUGE等指标
            return {"eval_loss": 0.0}
        else:
            # 对于分类任务
            predictions = np.argmax(predictions, axis=1)
            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions, average="weighted")
            }
    
    def train(self, train_data: List[Dict], val_data: List[Dict], 
              resume_from_checkpoint: str = None) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            train_data: 训练数据
            val_data: 验证数据
            resume_from_checkpoint: 检查点路径
            
        Returns:
            训练结果
        """
        # 加载模型和分词器
        if self.model is None:
            self.load_model_and_tokenizer()
            
        # 准备数据集
        train_dataset, val_dataset = self.prepare_datasets(train_data, val_data)
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_strategy=self.config.save_strategy,
            evaluation_strategy=self.config.evaluation_strategy,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # 禁用wandb等报告
            dataloader_pin_memory=False,
        )
        
        # 设置数据整理器
        if self.config.model_type == "seq2seq":
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True
            )
        else:
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # 创建训练器
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # 开始训练
        logging.info("开始训练模型...")
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # 保存模型
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # 保存训练信息
        train_info = {
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "train_loss": train_result.metrics.get("train_loss", 0),
            "config": self.config.__dict__,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(self.config.output_dir, "train_info.json"), "w", encoding="utf-8") as f:
            json.dump(train_info, f, ensure_ascii=False, indent=2)
            
        logging.info("模型训练完成")
        return train_info
    
    def evaluate(self, eval_data: List[Dict]) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            eval_data: 评估数据
            
        Returns:
            评估结果
        """
        if self.trainer is None:
            raise ValueError("模型尚未训练，请先调用train方法")
            
        eval_dataset = MedicalQADataset(eval_data, self.tokenizer, self.config)
        eval_result = self.trainer.evaluate(eval_dataset=eval_dataset)
        
        logging.info(f"模型评估结果: {eval_result}")
        return eval_result
    
    def save_model(self, save_path: str = None):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        if save_path is None:
            save_path = self.config.output_dir
            
        os.makedirs(save_path, exist_ok=True)
        
        if self.model is not None:
            self.model.save_pretrained(save_path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)
            
        # 保存配置
        config_path = os.path.join(save_path, "model_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.__dict__, f, ensure_ascii=False, indent=2)
            
        logging.info(f"模型已保存到: {save_path}")
    
    def load_trained_model(self, model_path: str):
        """
        加载已训练的模型
        
        Args:
            model_path: 模型路径
        """
        try:
            # 加载配置
            config_path = os.path.join(model_path, "model_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
                    for key, value in config_dict.items():
                        setattr(self.config, key, value)
            
            # 加载模型和分词器
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if self.config.model_type == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            else:
                self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
                
            self.model.to(self.device)
            logging.info(f"成功加载训练后的模型: {model_path}")
            
        except Exception as e:
            logging.error(f"加载训练后的模型失败: {str(e)}")
            raise
    
    def predict(self, question: str, context: str = "") -> str:
        """
        使用训练后的模型进行预测
        
        Args:
            question: 问题
            context: 上下文
            
        Returns:
            预测答案
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型尚未加载，请先训练或加载模型")
        
        # 准备输入
        if self.config.model_type == "seq2seq":
            input_text = f"问题: {question}"
            if context:
                input_text = f"背景: {context} 问题: {question}"
                
            # 编码输入
            inputs = self.tokenizer(
                input_text,
                max_length=self.config.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # 生成答案
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.config.max_target_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.7
                )
                
            # 解码答案
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer.strip()
            
        else:
            # 抽取式问答 (暂不实现)
            raise NotImplementedError("Extractive QA prediction not implemented yet")

def create_default_trainer() -> MedicalQATrainer:
    """创建默认配置的训练器"""
    config = ModelConfig(
        model_name="hfl/chinese-bert-wwm-ext",
        model_type="seq2seq",
        max_length=512,
        max_target_length=128,
        learning_rate=2e-5,
        batch_size=4,  # 减小batch size以适应内存限制
        num_epochs=3,
        output_dir="models/medical_qa"
    )
    
    return MedicalQATrainer(config)