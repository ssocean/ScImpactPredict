from transformers import pipeline
import torch
from datetime import datetime
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, AdamW, RobertaForSequenceClassification, AutoTokenizer, \
    AutoModelForSequenceClassification, FlaxLlamaForCausalLM, LlamaForSequenceClassification
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import random_split
from tqdm import tqdm
import os
import argparse
import json
import torch
from peft import PeftModel,PeftModelForTokenClassification
import os
import torch.nn as nn

from tools.order_metrics import *
LlamaForSequenceClassification
import transformers.models.qwen2

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType



title = '''Revolutionizing Research Impact: A Novel LLM Approach for Newborn Article Prediction'''

abstract = '''
In an era where scholarly outputs are burgeoning, identifying seminal research swiftly and efficiently stands as a critical challenge. This study introduces a transformative approach using state-of-the-art Large Language Models (LLMs) to predict the potential impact of newly published articles based solely on their titles and abstracts. We present the Topic Normalized Citation Success Index for the Same Period (TNCSISP), an innovative metric that transcends traditional reliance on historical data, providing a time-normalized evaluation of an article's influence across diverse disciplines. Our method dramatically enhances the ability to discern high-impact research at its inception, irrespective of existing citation data, thereby facilitating a proactive identification of groundbreaking work amidst the daily flood of publications. This capability is particularly pivotal for automated scientific research systems, which can now prioritize and synthesize high-value research without human intervention. Furthermore, our approach democratizes academic evaluation by focusing on content inherent to the work itself, rather than external influence metrics. The implications of this technology extend beyond academic publishing to influence funding decisions, academic promotions, and the broader landscape of knowledge discovery. By empowering stakeholders to recognize and leverage influential research swiftly, this model sets the stage for a new era of accelerated and more equitable scientific advancement.'''
text = f'''Given a certain paper, Title: {title}\n Abstract: {abstract}. \n Predict its normalized academic impact (between 0 and 1):'''

from peft import AutoPeftModelForCausalLM,AutoPeftModelForSequenceClassification,AutoPeftModelForTokenClassification
from transformers import AutoTokenizer
import torch
model_pth = r"/home/u1120220285/ScitePredict/official_runs/mc_qwen7b_ep10/checkpoint-790"
model = AutoPeftModelForSequenceClassification.from_pretrained(model_pth,num_labels=1, load_in_8bit=True,)
tokenizer = AutoTokenizer.from_pretrained(model_pth)

model = model.to("cuda")
model.eval()
titles = [
    "Predicting the Pulse of Research: Leveraging LLMs for Newborn Article Impact Analysis",
    "Words to Impact: Transforming Article Assessment with Large Language Models",
    "Future Insights: LLM-Driven Prediction of Scholarly Article Impact",
    "The Next Frontier in Research Evaluation: LLMs Predicting Newborn Article Influence",
    "From Publication to Prediction: LLMs Forecasting the Future of Scholarly Articles",
    "Innovating Impact: How LLMs Revolutionize the Prediction of Research Relevance",
    "Charting New Territories: Using LLMs to Assess the Immediate Impact of Scholarly Articles",
    "The Predictive Scholar: LLMs as a Tool for Evaluating Emerging Research",
    "LLM Insight: A New Model for Predicting Scholarly Impact at Publication",
    "Forecasting the Academic Frontier: LLMs and the Prediction of Article Impact"
]
titles = [
    "Shaping the Future of Scholarship: Predictive Insights with Large Language Models",
    "Emerging Research Frontiers: Advanced LLMs for Predicting Scholarly Impact",
    "Revolutionizing Academic Discovery with Predictive Language Modeling",
    "Beyond Citations: A New Paradigm for Measuring Scholarly Influence with LLMs",
    "Predictive Analytics in Scholarship: The Power of LLMs in Impact Assessment",
    "Decoding Academic Futures: Large Language Models as Predictive Tools",
    "The New Vanguard of Research: Predicting Academic Impact with LLMs",
    "Navigating the Scholarly Landscape: How LLMs Forecast Academic Trends",
    "Illuminating Research Trajectories with Language Model Predictions",
    "From Text to Trajectory: Using LLMs to Foresee Academic Impact",
    "Unlocking the Potential of New Research with Language Models",
    "Assessing the Unseen: Predicting the Reach of Scholarly Work with LLMs",
    "Forecasting Intellectual Milestones with Advanced Language Modeling",
    "Redefining Research Evaluation through Predictive Language Models",
    "Language Models as Beacons of Academic Progress",
    "Envisioning the Impact of New Scholarships through LLMs",
    "The Predictive Scholar: Harnessing LLMs for Insightful Academic Analysis",
    "Language Models: The New Predictors of Scholarly Success",
    "Charting New Academic Realms with Predictive Language Technologies",
    "Transforming How We Measure Research Impact with LLM Insights"
]
title = 'From Words to Worth: Newborn Article Impact Prediction with Large Language Models'
abstracts = ['''In an era where scholarly outputs are burgeoning, identifying seminal research swiftly and efficiently stands as a critical challenge. This study introduces a transformative approach using state-of-the-art Large Language Models (LLMs) to predict the potential impact of newly published articles based solely on their titles and abstracts. We present the Topic Normalized Citation Success Index for the Same Period (TNCSISP), an innovative metric that transcends traditional reliance on historical data, providing a time-normalized evaluation of an article's influence across diverse disciplines. Our method dramatically enhances the ability to discern high-impact research at its inception, irrespective of existing citation data, thereby facilitating a proactive identification of groundbreaking work amidst the daily flood of publications. This capability is particularly pivotal for automated scientific research systems, which can now prioritize and synthesize high-value research without human intervention. Furthermore, our approach democratizes academic evaluation by focusing on content inherent to the work itself, rather than external influence metrics. The implications of this technology extend beyond academic publishing to influence funding decisions, academic promotions, and the broader landscape of knowledge discovery. By empowering stakeholders to recognize and leverage influential research swiftly, this model sets the stage for a new era of accelerated and more equitable scientific advancement.''',
             

]
for abstract in abstracts:

    # text = text = f"""
    # Given a certain paper, Title: {title}\n Abstract: {abstract}\n State-of-the-Art Performance: 'Yes'. \n Released a New Dataset: 'Yes'. \n Code Open Access: 'Yes'.\n Reference Quality Metric(on a scale from lowest 0 to highest 1): {0.98} . \n Predict its normalized academic impact (between 0 and 1):
    # """
    text = f'''Given a certain paper, Title: {title}\n Abstract: {abstract}. \n Predict its normalized academic impact (between 0 and 1):'''
    inputs = tokenizer(text, return_tensors="pt")

    outputs = model(input_ids=inputs["input_ids"].to("cuda"))
    print(outputs['logits'])

# "Preheat the oven to 350 degrees and place the cookie dough in the center of the oven. In a large bowl, combine the flour, baking powder, baking soda, salt, and cinnamon. In a separate bowl, combine the egg yolks, sugar, and vanilla."
# from transformers import AutoModel, AutoTokenizer

# # 模型和分词器的路径
# model_path = "/home/u1120220285/ScitePredict/official_runs/mc_LLAMA3/checkpoint-395"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
    
# tokenizer.pad_token = tokenizer.eos_token
# device_map={'':torch.cuda.current_device()}

# model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1, load_in_8bit=True,device_map=device_map,)
# model.config.pad_token_id = model.config.eos_token_id
# print(model.score.weight)
# model.score.load_state_dict(torch.load(os.path.join(model_path,'score.pt')))
# print(model.score.weight)
# # 准备输入数据
# inputs = tokenizer(text, return_tensors='pt')

# model.eval()  # 设置为评估模式
# with torch.no_grad():
#     outputs = model(**inputs)

# # 提取 logits 并计算概率
# logits = outputs.logits

# print("Logits:", logits)


# from evaluate import evaluator
# from transformers import AutoModelForSequenceClassification, pipeline

# data = load_dataset("imdb", split="test").shuffle(seed=42).select(range(1000))
# task_evaluator = evaluator("text-classification")

# class TextDataset(Dataset):
#     def __init__(self, data, tokenizer, max_length=512):
#         self.data = data
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         label = float(row['TNCSI_SP'])
#         if args.data_style == 0:
#             text = f'''Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}. \n Predict its normalized academic impact (between 0 and 1):'''
#             inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
#                                     return_tensors="pt")
#             return {
#                 'input_ids': inputs['input_ids'].squeeze(0),
#                 'attention_mask': inputs['attention_mask'].squeeze(0),
#                 'labels': torch.tensor(label, dtype=torch.float)
#             }
# full_data = pd.read_csv(r'/home/u1120220285/ScitePredict/NAID/NAID_test_extrainfo.csv')
# dataset = TextDataset(full_data, tokenizer, max_length=args.max_length)

# test_size = int(args.test_ratio * len(dataset)) 
# train_size = len(dataset) -test_size



# test_dataset,train_dataset = random_split(dataset, [ test_size,train_size])


# # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
# # 1. Pass a model name or path
# eval_results = task_evaluator.compute(
#     model_or_pipeline="/home/u1120220285/ScitePredict/official_runs/mc_LLAMA3/checkpoint-395",
#     data=test_dataset,
#     input_column="text",
#     label_column="label"
# )

# eval_results = task_evaluator.compute(
#     model_or_pipeline=pipe,
#     data=data,
#     label_mapping={"NEGATIVE": 0, "POSITIVE": 1}
# )
# print(eval_results)