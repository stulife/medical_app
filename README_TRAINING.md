# 医疗问答模型训练和自主学习系统

## 概述

本系统为您的医疗问答应用增加了以下功能：
1. **自主训练模型** - 基于Transformers的深度学习问答模型
2. **自主学习机制** - 基于用户反馈的持续改进
3. **模型评估** - 全面的性能评估指标
4. **训练数据管理** - 智能数据预处理和管理

## 系统架构

```
医疗问答系统
├── 基础问答引擎 (api/models/medical_model.py)
├── 深度学习训练器 (api/models/qa_trainer.py)
├── 数据管理器 (api/utils/training_data_manager.py)
├── 模型评估器 (api/utils/model_evaluator.py)
├── 自主学习管理器 (api/utils/auto_learning.py)
├── 训练脚本 (train.py)
└── 配置文件 (config/training_config.json)
```

## 功能特性

### 🤖 自主训练模型
- 基于Transformers的中文医疗问答模型
- 支持生成式问答 (Seq2Seq)
- 可配置的训练参数
- 模型版本管理和备份

### 📊 智能数据处理
- 自动从知识库生成问答对 (182个训练样本)
- 支持用户反馈数据集成
- 数据质量检查和去重
- 训练/验证集自动划分

### 🔄 自主学习机制
- 基于用户反馈的自动重训练
- 可配置的触发条件
- 模型性能监控
- 增量学习支持

### 📈 模型评估
- 多维度评估指标 (F1, BLEU, ROUGE, 语义相似度)
- 医疗准确性评估
- 批量评估和报告生成
- 性能改进追踪

## 快速开始

### 1. 环境准备

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装深度学习依赖（可选）
pip install torch==2.0.1 transformers==4.33.0 accelerate==0.21.0
```

### 2. 基础功能测试

```bash
# 运行基础功能测试
python test_basic.py

# 运行完整测试（需要深度学习依赖）
python test_training.py
```

### 3. 数据准备

```bash
# 生成训练数据
python train.py prepare --include-feedback --output-dir api/data

# 查看数据统计
python train.py status
```

### 4. 模型训练

```bash
# 使用默认配置训练
python train.py train

# 使用自定义配置训练
python train.py train --config config/training_config.json --force

# 从检查点恢复训练
python train.py train --resume-from-checkpoint models/medical_qa/checkpoint-100
```

### 5. 模型评估

```bash
# 评估训练后的模型
python train.py evaluate --model-path models/medical_qa

# 使用自定义测试数据评估
python train.py evaluate --test-file my_test_data.json --output-dir results
```

### 6. 自主学习

```bash
# 启动自主学习
python train.py auto-learn

# 强制重训练
python train.py auto-learn --force-retrain

# 使用自定义配置
python train.py auto-learn --config config/training_config.json
```

## 配置说明

### 训练配置 (config/training_config.json)

```json
{
  "model_name": "hfl/chinese-bert-wwm-ext",  // 预训练模型
  "model_type": "seq2seq",                   // 模型类型
  "max_length": 512,                         // 最大输入长度
  "max_target_length": 128,                  // 最大输出长度
  "learning_rate": 2e-5,                     // 学习率
  "batch_size": 4,                           // 批次大小
  "num_epochs": 3,                           // 训练轮数
  "output_dir": "models/medical_qa"          // 模型保存路径
}
```

### 自主学习配置

```json
{
  "auto_learning": {
    "min_feedback_count": 10,          // 最小反馈数量阈值
    "min_low_score_ratio": 0.3,        // 低分反馈比例阈值
    "feedback_score_threshold": 0.6,   // 反馈评分阈值
    "retrain_interval_hours": 24,      // 重训练间隔(小时)
    "max_training_samples": 1000       // 最大训练样本数
  }
}
```

## API 使用示例

### 基础问答 (不需要深度学习依赖)

```python
from api.models.medical_model import MedicalQAModel

# 创建模型实例
model = MedicalQAModel(use_deep_learning=False)

# 生成答案
result = model.generate_answer("感冒有什么症状？")
print(result['answer'])
print(f"来源: {result['source']}")
print(f"置信度: {result['confidence']}")
```

### 深度学习问答

```python
from api.models.medical_model import MedicalQAModel

# 创建启用深度学习的模型
model = MedicalQAModel(use_deep_learning=True)

# 生成答案
result = model.generate_answer(
    question="高血压用什么药治疗？",
    context="患者询问药物治疗",
    session_id="user_123"
)

print(result['answer'])
```

### 添加用户反馈

```python
# 添加用户反馈用于自主学习
model.add_feedback(
    question="感冒症状是什么？",
    predicted_answer="感冒有发热症状",
    correct_answer="感冒的症状包括发热、咳嗽、流鼻涕、头痛、乏力等",
    score=0.6,
    feedback="答案不够完整"
)
```

### 模型训练

```python
from api.models.qa_trainer import MedicalQATrainer, ModelConfig
from api.utils.training_data_manager import TrainingDataManager

# 准备训练数据
data_manager = TrainingDataManager('api/data')
train_data, val_data = data_manager.prepare_training_data()

# 配置训练参数
config = ModelConfig(
    model_name="hfl/chinese-bert-wwm-ext",
    num_epochs=2,
    batch_size=4
)

# 创建训练器并训练
trainer = MedicalQATrainer(config)
trainer.train(train_data, val_data)
```

### 模型评估

```python
from api.utils.model_evaluator import MedicalQAEvaluator

evaluator = MedicalQAEvaluator()

predictions = ["感冒的症状包括发热、咳嗽", "建议前往内科就诊"]
references = ["感冒症状：发热、咳嗽、流鼻涕", "建议挂内科号"]

result = evaluator.evaluate_batch(predictions, references)
print(result['overall_metrics'])
```

## 性能指标说明

- **F1 Score**: 精确率和召回率的调和平均
- **Exact Match**: 完全匹配率
- **BLEU Score**: 机器翻译评估指标
- **ROUGE Score**: 文本摘要评估指标
- **语义相似度**: 基于序列匹配的相似度
- **医疗准确性**: 医疗实体匹配准确性

## 故障排除

### 1. 依赖问题
```bash
# 如果torch版本冲突
pip uninstall torch transformers accelerate
pip install torch==2.0.1 transformers==4.33.0 accelerate==0.21.0
```

### 2. 内存不足
- 减小batch_size (如设为2)
- 减小max_length (如设为256)
- 减少训练样本数

### 3. 模型加载失败
- 检查模型路径是否正确
- 确保有足够的磁盘空间
- 检查网络连接（首次下载预训练模型时）

### 4. 训练数据不足
```bash
# 强制训练（即使数据不足）
python train.py train --force
```

## 文件结构

```
medical_app/
├── api/
│   ├── data/                          # 数据文件
│   │   ├── medical_data.json         # 基础医疗数据
│   │   ├── knowledge_graph.json      # 知识图谱
│   │   ├── feedback_data.json        # 用户反馈数据
│   │   ├── train_data.json          # 训练数据
│   │   └── val_data.json            # 验证数据
│   ├── models/                       # 模型文件
│   │   ├── medical_model.py         # 主要医疗模型
│   │   └── qa_trainer.py            # 训练器
│   └── utils/                       # 工具模块
│       ├── training_data_manager.py # 数据管理器
│       ├── model_evaluator.py       # 模型评估器
│       └── auto_learning.py         # 自主学习管理器
├── models/                          # 训练后的模型
│   └── medical_qa/                  # 医疗问答模型
├── config/                          # 配置文件
│   └── training_config.json         # 训练配置
├── train.py                         # 训练脚本
├── test_basic.py                    # 基础功能测试
├── test_training.py                 # 完整功能测试
└── requirements.txt                 # 依赖列表
```

## 高级功能

### 1. 自定义模型
可以通过修改`config/training_config.json`来使用不同的预训练模型：
- `hfl/chinese-bert-wwm-ext` (推荐，中文优化)
- `bert-base-chinese`
- `uer/chinese_roberta_L-12_H-768`

### 2. 分布式训练
对于大规模数据，可以配置多GPU训练：
```json
{
  "training_args": {
    "dataloader_num_workers": 4,
    "ddp_backend": "nccl"
  }
}
```

### 3. 模型量化
为了减少模型大小和推理时间：
```json
{
  "deployment": {
    "model_optimization": {
      "use_quantization": true,
      "use_onnx": true
    }
  }
}
```

## 许可证

本项目遵循原项目的许可证条款。

## 支持

如有问题，请查看：
1. 测试日志: `test_basic.log`, `test_training.log`
2. 训练日志: `training.log`
3. 运行基础测试: `python test_basic.py`