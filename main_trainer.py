import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import json
from datetime import datetime
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
# 加载和保存参数
def save_args_to_json(args, file_path):
    with open(file_path, 'w') as f:
        json.dump(args.to_dict(), f, indent=4)

# 命令行参数 (这里直接作为函数参数给出)
args = {
    "max_length": 1024,
    "batch_size": 8,
    "total_epochs": 100,
    "learning_rate": 5e-5,
    "warmup_ratio": 0.1,
    "data_path": '/home/u1120220285/ScitePredict/tools/Data_TNCSI_S.csv',  # 更新为你的数据路径
    "checkpoint": '/home/u1120220285/llama3_weight',  # 更新为你的模型检查点路径
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "num_labels": 1,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_bias": 'none',
    "target_modules": 'q_proj,v_proj,score',
    "runs_dir": './runs/' + datetime.now().strftime("%m-%d-%H-%M"),
}

# 创建输出目录
os.makedirs(args["runs_dir"], exist_ok=True)

# 加载数据集
data = pd.read_csv(args['data_path'])

class TextDataset(Dataset):
    def __init__(self, data, max_length):
        self.examples = []
        self.tokenizer = AutoTokenizer.from_pretrained(args['checkpoint'])
        for i, row in data.iterrows():
            text = f"Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}. \n Predict its normalized scholar impact (between 0 and 1):"
            inputs = self.tokenizer(text, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
            self.examples.append({
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(row['TNCSI'], dtype=torch.float)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


dataset = TextDataset(data, args['max_length'])

# 创建数据集分割
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# 定义 LoRA 配置
lora_config = LoraConfig(
    r=args['lora_r'],
    lora_alpha=args['lora_alpha'],
    lora_dropout=args['lora_dropout'],
    bias=args['lora_bias'],
    target_modules=args['target_modules'].split(','),
    task_type=TaskType.TOKEN_CLS
)

# 初始化模型并应用 LoRA
model = AutoModelForSequenceClassification.from_pretrained(args['checkpoint'], num_labels=args['num_labels'])
model = prepare_model_for_kbit_training(model)  # 准备8-bit训练
model = get_peft_model(model, lora_config)  # 应用 LoRA 配置

# TrainingArguments 设置
training_args = TrainingArguments(
    output_dir=args['runs_dir'],
    num_train_epochs=args['total_epochs'],
    per_device_train_batch_size=args['batch_size'],
    per_device_eval_batch_size=args['batch_size'],
    warmup_ratio=args['warmup_ratio'],
    learning_rate=args['learning_rate'],
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir=args['runs_dir'],
    logging_steps=10,
)

# 评估指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions[:, 0]
    return {
        "mse": ((predictions - labels) ** 2).mean()
    }

# 初始化模型
model = AutoModelForSequenceClassification.from_pretrained(args['checkpoint'], num_labels=args['num_labels'])
model.config.pad_token_id = model.config.eos_token_id


# import transformers.models.qwen2

# 
 
# # Define LoRA Config
# # Setup LoRA
# lora_config = LoraConfig(
#     r=args['lora_r'],
#     lora_alpha=args.lora_alpha,
#     lora_dropout=args.lora_dropout,
#     bias=args.lora_bias,
#     target_modules=args.target_modules.split(','),
#     task_type=TaskType.TOKEN_CLS
# )

# # prepare int-8 model for training
# model = prepare_model_for_kbit_training(model)

# # add LoRA adaptor
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 开始训练
trainer.train()

# 保存模型
model_path = os.path.join(args['runs_dir'], 'best_model')
model.save_pretrained(model_path)
# tokenizer.save_pretrained(model_path)

# 输出训练参数和路径
save_args_to_json(training_args, os.path.join(model_path, 'training_args.json'))
print(f"Model and training logs saved to {model_path}")
