from sklearn.metrics import ndcg_score
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from accelerate import Accelerator
import pandas as pd
from datetime import datetime
import logging
import random
import time
import torch
from torch.nn.functional import mse_loss,l1_loss
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, AdamW, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, FlaxLlamaForCausalLM
from torch.utils.data import random_split
from transformers import LlamaForSequenceClassification, GemmaForSequenceClassification,MistralForSequenceClassification,Qwen2ForSequenceClassification,Phi3ForSequenceClassification
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import random_split
from tqdm import tqdm
import os
import argparse
import json
from accelerate import Accelerator
import os
import torch.nn as nn
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import transformers.models.qwen2
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
accelerator = Accelerator()

def NDCG_k(predictions, labels, k=20):
    if len(predictions) < k:
        return -1  # or handle as preferred
    return ndcg_score([labels], [predictions], k=k)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.tensor(predictions).squeeze()
    labels = torch.tensor(labels).squeeze()
    mse = nn.MSELoss()(predictions, labels).item()
    mae = nn.L1Loss()(predictions, labels).item()
    
    # Convert tensors to numpy arrays for NDCG computation
    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    # Calculate NDCG
    ndcg = NDCG_k(predictions, labels)
    
    
    return {"mse": mse, "mae": mae, "ndcg": ndcg}
class TextDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=1024,prompt_style=0):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.prompt_style = prompt_style
        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            label = float(row['TNCSI_SP'])
            if self.prompt_style == 1:
                text = f'''Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}. \n Predict its normalized academic impact:'''
                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 0:
                text = f'''Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}. \n Predict its normalized academic impact (between 0 and 1):'''
                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == -3:
                label = float(row['TNCSI'])
                text = f'''Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}. \n Predict its normalized academic impact (between 0 and 1):'''
                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }

            elif self.prompt_style == 101:
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                OA = row['OA']
                RQM = round(row['RQM'],1)

                dataset_statement = "This paper has released a new dataset." if new_dataset else "No new dataset has been released."

                oa_statement = 'This paper has open-accessed the code framework.' if OA else "The code is not available."
                
                if SOTA == 1:
                    sota_statement = 'True'
                else:
                    sota_statement = 'False'
                text = text = f"""
                Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}\n State-of-the-Art Performance: {'Yes' if SOTA == 1 else 'No'}. \n Predict its normalized academic impact (between 0 and 1):
                """
                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                token_count = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum().item()
                if token_count > self.max_length:
                    print("The text has been truncated.")
      
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 102:
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                OA = row['OA']
                RQM = round(row['RQM'],1)

                dataset_statement = "This paper has released a new dataset." if new_dataset else "No new dataset has been released."

                oa_statement = 'This paper has open-accessed the code framework.' if OA else "The code is not available."
                
                if SOTA == 1:
                    sota_statement = 'True'
                else:
                    sota_statement = 'False'
                # - State-of-the-Art Performance: {'Yes' if SOTA == 1 else 'No'}
                # - Released a New Dataset: {'Yes' if new_dataset else 'No'}
                # - Code Available Open Access: {'Yes' if OA else 'No'}
                # - Reference Quality Metric: {RQM} (on a scale from lowest 0 to highest 1)

                text = text = f"""
                Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}\n Released a New Dataset: {'Yes' if new_dataset else 'No'}. \n Predict its normalized academic impact (between 0 and 1):
                """
                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                token_count = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum().item()
                if token_count > self.max_length:
                    print("The text has been truncated.")
      
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 103:
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                OA = row['OA']
                RQM = round(row['RQM'],1)

                dataset_statement = "This paper has released a new dataset." if new_dataset else "No new dataset has been released."

                oa_statement = 'This paper has open-accessed the code framework.' if OA else "The code is not available."
                
                if SOTA == 1:
                    sota_statement = 'True'
                else:
                    sota_statement = 'False'
                # - State-of-the-Art Performance: {'Yes' if SOTA == 1 else 'No'}
                # - Released a New Dataset: {'Yes' if new_dataset else 'No'}
                # - Code Available Open Access: {'Yes' if OA else 'No'}
                # - Reference Quality Metric(on a scale from lowest 0 to highest 1): {RQM} 

                text = text = f"""
                Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}\n Code Open Access: {'Yes' if OA else 'No'}. \n Predict its normalized academic impact (between 0 and 1):
                """
                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                token_count = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum().item()
                if token_count > self.max_length:
                    print("The text has been truncated.")
      
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 104:
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                OA = row['OA']
                RQM = round(row['RQM'],1)

                dataset_statement = "This paper has released a new dataset." if new_dataset else "No new dataset has been released."

                oa_statement = 'This paper has open-accessed the code framework.' if OA else "The code is not available."
                
                if SOTA == 1:
                    sota_statement = 'True'
                else:
                    sota_statement = 'False'
                # - State-of-the-Art Performance: {'Yes' if SOTA == 1 else 'No'}
                # - Released a New Dataset: {'Yes' if new_dataset else 'No'}
                # - Code Available Open Access: {'Yes' if OA else 'No'}
                # - Reference Quality Metric: {RQM} (on a scale from lowest 0 to highest 1)

                text = text = f"""
                Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}\n Reference Quality Metric(on a scale from lowest 0 to highest 1): {RQM} . \n Predict its normalized academic impact (between 0 and 1):
                """
                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                token_count = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum().item()
                if token_count > self.max_length:
                    print("The text has been truncated.")
      
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 105:
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                OA = row['OA']
                RQM = round(row['RQM'],1)

                dataset_statement = "This paper has released a new dataset." if new_dataset else "No new dataset has been released."

                oa_statement = 'This paper has open-accessed the code framework.' if OA else "The code is not available."
                
                if SOTA == 1:
                    sota_statement = 'True'
                else:
                    sota_statement = 'False'
                # - State-of-the-Art Performance: {'Yes' if SOTA == 1 else 'No'}
                # - Released a New Dataset: {'Yes' if new_dataset else 'No'}
                # - Code Available Open Access: {'Yes' if OA else 'No'}
                # - Reference Quality Metric: {RQM} (on a scale from lowest 0 to highest 1)

                text = text = f"""
                Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}\n State-of-the-Art Performance: {'Yes' if SOTA == 1 else 'No'}. \n Released a New Dataset: {'Yes' if new_dataset else 'No'} \n Code Open Access: {'Yes' if OA else 'No'}.\n Reference Quality Metric(on a scale from lowest 0 to highest 1): {RQM} . \n Predict its normalized academic impact (between 0 and 1):
                """
                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                token_count = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum().item()
                if token_count > self.max_length:
                    print("The text has been truncated.")
      
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }              
            elif self.prompt_style == 2:
                text = f'''Given the provided title and abstract, predict the future normalized academic impact on a scale from 0 (lowest impact) to 1 (highest impact). You may consider factors such as the language clarity, novelty of the research, or the claim of state-of-the-art, etc. Title: {row['title']}\n Abstract: {row['abstract']}'''
                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 3:
                text = f'''Title: {row['title']}\n Abstract: {row['abstract']}'''
                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 4:
                is_practical = row['is_practical']
                new_task = row['new_task']
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                is_broad = row['is_broad']

                # Preparing statements based on the values
                practical_statement = "This paper is a practical engineering paper." if is_practical else "This paper is a theoretical paper."
                task_statement = "This paper introduces a new task." if new_task else "This paper does not introduce a new task."
                dataset_statement = "This paper contributes a new dataset." if new_dataset else "This paper does not contribute a new dataset."
                sota_statement = "This paper claims state-of-the-art performance." if SOTA == 1 else "This paper does not claim state-of-the-art performance." if SOTA == 0 else "State-of-the-art performance is not applicable."
                broad_statement = "This work is aimed at tasks in a general broad field." if is_broad else "This work focuses on a specific subfield."

                # Constructing the final text
                text = f'''
                Given a certain paper, Title: {row['title']}\n
                Abstract: {row['abstract']}\n
                Additional Status Information:\n
                {practical_statement}\n
                {task_statement}\n
                {dataset_statement}\n
                {sota_statement}\n
                {broad_statement}\n
                Predict its normalized academic impact (between 0 and 1):
                '''
                # print(text)
                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 5:
                is_practical = row['is_practical']
                new_task = row['new_task']
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                is_broad = row['is_broad']
                OA = row['OA']
                # Preparing statements based on the values
                practical_statement = "Practical engineering paper." if is_practical else "Theoretical paper."
                task_statement = "Introduces new task." if new_task else "No new task."
                dataset_statement = "Contributes new dataset." if new_dataset else "No new dataset."
                sota_statement = "Claims state-of-the-art performance." if SOTA == 1 else "No state-of-the-art claim." if SOTA == 0 else "State-of-the-art not applicable."
                broad_statement = "Aimed at general field." if is_broad else "Focuses on specific subfield."
                oa_statement = 'Code is publicly avaiable.' if OA else "Code is unavaiable."
                # Constructing the final text
                text = f'''
                Given a certain paper with the following details:
                Title: {row['title']}
                Abstract: {row['abstract']}
                Paper Type: {practical_statement}
                Task Status: {task_statement}
                Dataset Status:{dataset_statement}
                State-of-the-art Status: {sota_statement}
                Open-access Code Status: {oa_statement}
                Predict the probability (between 0 and 1) that this paper will be cited more times than other papers in the same field within the same year. You must reply with only one number.
                '''
                # print(text)
                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 6:
                is_practical = row['is_practical']
                new_task = row['new_task']
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                is_broad = row['is_broad']
                OA = row['OA']
                # Preparing statements based on the values
                practical_statement = "Engineering Paper." if is_practical else "Theoretical Paper."
                task_statement = "Yes" if new_task else "No"
                dataset_statement = "Yes" if new_dataset else "No"
                sota_statement = "Yes" if SOTA == 1 else "No" if SOTA == 0 else "Not Applicable."
                broad_statement = "General Field" if is_broad else "Specific Subfield."
                oa_statement = 'Yes' if OA else "No"
                # Constructing the final text
                text = f'''
                Given an academic paper with the following information:
                Title: {row['title']}
                Abstract: {row['abstract']}
                Addition Information:
                Paper Type: {practical_statement}
                Discover New Task?: {task_statement} || Release Dataset?: {dataset_statement} || Achieve the State-of-the-art Performance?: {sota_statement} || Open-access Code?: {oa_statement}
                Based on the above information, predict the probability (between 0 and 1) that this paper will be cited more times than other papers in the same field within the same year. You must reply with only one number.
                '''

                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 7:
                is_practical = row['is_practical']
                new_task = row['new_task']
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                is_broad = row['is_broad']
                OA = row['OA']
                RQM = round(row['RQM'], 1)
                SMP = int(row['SMP'])
                # Preparing statements based on the values
                practical_statement = "Engineering Paper." if is_practical else "Theoretical Paper."
                task_statement = "Yes" if new_task else "No"
                dataset_statement = "Yes" if new_dataset else "No"
                sota_statement = "Yes" if SOTA == 1 else "No" if SOTA == 0 else "Not Applicable."
                broad_statement = "General Field" if is_broad else "Specific Subfield."
                oa_statement = 'Yes' if OA else "No"
                # Constructing the final text
                text = f'''
                Given an academic paper with the following information:
                Title: {row['title']}
                Abstract: {row['abstract']}
                Addition Information:
                Paper Type: {practical_statement}
                New Task?: {task_statement} || Release Dataset?: {dataset_statement} || Achieve the State-of-the-art Performance?: {sota_statement} || Open-access Code?: {oa_statement}
                Reference Quaility, from 0 low to 1 high: {RQM}
                Median Publication Age of ReferenceL {SMP} months
                Based on the above information, predict the probability (between 0 and 1) that this paper will be cited more times than other papers in the same field within the same year. You must reply with only one number.
                '''

                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                token_count = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum().item()
                if token_count > self.max_length:
                    print("The text has been truncated.")
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 8:
                is_practical = row['is_practical']
                new_task = row['new_task']
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                is_broad = row['is_broad']
                OA = row['OA']
                RQM = round(row['RQM'], 1)
                SMP = int(row['SMP'])
                # Preparing statements based on the values
                practical_statement = "This paper is a practical engineering paper, " if is_practical else "This paper is a theoretical paper, "
                task_statement = "which introduces a new task. " if new_task else "which focus on the existing task. "
                dataset_statement = "A new dataset has been constructed and released." if new_dataset else "No new dataset has been constructed and released. "
                broad_statement = "broad" if is_broad else "specific"
                oa_statement = 'is open-accessed' if OA else "unavailable"
                if SOTA == 1:
                    sota_statement = f"The method achieve the SOTA performance in a {broad_statement} field, and the related code {oa_statement}. "
                else:
                    sota_statement =f"The method focus on a {broad_statement} field, and the related code {oa_statement}. " #if SOTA == 0 else "State-of-the-art performance is not applicable."
                ref_statement = f'The average quaility of referenced literature is {RQM} (from lowest 0 to highest 1), and the median publication age of references is {SMP} months. '
                
                # Constructing the final text
                text = f'''
                You are an experienced journal editor, skilled at predicting the impact of articles. Here is the detailed information of a specific article.
                Title: {row['title']}
                Abstract: {row['abstract']}
                Preliminary review:
                {practical_statement}{task_statement}{dataset_statement}{sota_statement}{ref_statement}
                Carefully read the title, abstract, and preliminary review, then predict its normalized academic impact (between 0 and 1):
                '''

                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                token_count = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum().item()
                if token_count > self.max_length:
                    print("The text has been truncated.")
      
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 9:
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                is_broad = row['is_broad']
                OA = row['OA']
                if row['RQM'] >0.7:
                    RQM = 'high'
                elif row['RQM'] >0.4:
                    RQM = 'medium'
                else:
                    RQM = 'low'
                SMP = int(row['SMP'])
 
                dataset_statement = "A new dataset has been constructed and released." if new_dataset else "No new dataset has been constructed and released. "
                broad_statement = "broad" if is_broad else "specific"
                oa_statement = 'is open-accessed' if OA else "unavailable"
                if SOTA == 1:
                    sota_statement = f"The method achieve the SOTA performance in a {broad_statement} field, and the related code {oa_statement}. "
                else:
                    sota_statement =f"The method focus on a {broad_statement} field, and the related code {oa_statement}. " #if SOTA == 0 else "State-of-the-art performance is not applicable."
                ref_statement = f'The average quaility of referenced papers is {RQM}, and the median publication age of references is {SMP} months. '
                
                # Constructing the final text
                text = f'''
                You are an experienced journal editor, skilled at predicting the impact of articles. Here is the detailed information of a specific article.
                Title: {row['title']}
                Abstract: {row['abstract']}
                Preliminary review:
                {dataset_statement}{sota_statement}{ref_statement}
                Carefully read the title, abstract, and preliminary review, then predict its normalized academic impact (between 0 and 1):
                '''

                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                token_count = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum().item()
                if token_count > self.max_length:
                    print("The text has been truncated.")
      
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 91:
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                OA = row['OA']
                dataset_statement = "A new dataset has been constructed and released." if new_dataset else "No new dataset has been constructed and released. "
                oa_statement = 'and the related code is open-accessed' if OA else "but the related code is unavailable"
                if SOTA == 1:
                    sota_statement = f"The method achieve the SOTA performance, {oa_statement}. "
                else:
                    sota_statement = 'The related code is open-accessed. ' if OA else "The related code is unavailable. "#if SOTA == 0 else "State-of-the-art performance is not applicable."
                ref_statement = f'The average quaility index of referenced papers is {row["RQM"]} (0 refers to the lowest and 1 refers to the highest).'
                
                # Constructing the final text
                text = f'''
                You are an experienced journal editor, skilled at predicting the impact of articles. Here is the detailed information of a specific article.
                Title: {row['title']}
                Abstract: {row['abstract']}
                Preliminary review:
                {dataset_statement}{sota_statement}{ref_statement}
                Carefully read the title, abstract, and preliminary review, then predict its normalized academic impact (between lowest 0 and highest 1):
                '''

                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                token_count = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum().item()
                if token_count > self.max_length:
                    print("The text has been truncated.")
      
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 10:
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                is_broad = row['is_broad']
                OA = row['OA']
                RQM = round(row['RQM'],1)
                # if row['RQM'] >0.7:
                #     RQM = 'high'
                # elif row['RQM'] >0.4:
                #     RQM = 'medium'
                # else:
                #     RQM = 'low'
                SMP = int(row['SMP'])
 
                dataset_statement = "True" if new_dataset else "False"

                oa_statement = 'True' if OA else "False"
                if SOTA == 1:
                    sota_statement = 'True'
                else:
                    sota_statement = 'False'
                
                
                # Constructing the final text
                text = f'''
                Given a certain paper,
                <Title>: {row['title']}
                <Abstract>: {row['abstract']}
                <Release New Dataset>: {dataset_statement}
                <Open-Access Code>: {oa_statement}
                <Reference Quaility (form 0-1)>: {RQM}
                <Reference Timeliness (the lower the better)>: {SMP} months
                Predict its normalized academic impact (between 0 and 1):
                '''

                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                token_count = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum().item()
                if token_count > self.max_length:
                    print("The text has been truncated.")
      
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 11:
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                OA = row['OA']
                RQM = round(row['RQM'],1)
                SMP = int(row['SMP'])
 
                dataset_statement = "True" if new_dataset else "False"

                oa_statement = 'True' if OA else "False"
                if SOTA == 1:
                    sota_statement = 'True'
                else:
                    sota_statement = 'False'
                
                
                # Constructing the final text
                text = f'''
                Given a certain paper,
                <Title>: {row['title']}
                <Abstract>: {row['abstract']}
                <Achieve SOTA Performance>: {SOTA}
                <Release New Dataset>: {dataset_statement}
                <Open-Access Code>: {oa_statement}
                <Reference Quaility (between 0 and 1)>: {RQM}
                Predict its normalized academic impact (between 0 and 1):
                '''

                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                token_count = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum().item()
                if token_count > self.max_length:
                    print("The text has been truncated.")
      
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 12:
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                OA = row['OA']
                RQM = round(row['RQM'],1)

 
                dataset_statement = "This paper has released a new dataset." if new_dataset else "No new dataset has been released."

                oa_statement = 'This paper has open-accessed the code framework.' if OA else "The code is not available."
                if SOTA == 1:
                    sota_statement = 'True'
                else:
                    sota_statement = 'False'
                
                
                # Constructing the final text
                text = text = f"""
                You are an experienced journal editor, assess the future academic impact of the following research paper:
                - Title: {row['title']}
                - Abstract: {row['abstract']}
                - State-of-the-Art Performance: {'Yes' if SOTA == 1 else 'No'}
                - Released a New Dataset: {'Yes' if new_dataset else 'No'}
                - Code Available Open Access: {'Yes' if OA else 'No'}
                - Reference Quality Metric: {RQM} 

                Based on the information provided, predict the future normalized academic impact of this paper, on a scale from lowest 0 to highest 1.
                """


                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                token_count = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum().item()
                if token_count > self.max_length:
                    print("The text has been truncated.")
      
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 13:
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                OA = row['OA']
                RQM = round(row['RQM'],1)

                dataset_statement = "This paper has released a new dataset." if new_dataset else "No new dataset has been released."

                oa_statement = 'This paper has open-accessed the code framework.' if OA else "The code is not available."
                if SOTA == 1:
                    sota_statement = 'True'
                else:
                    sota_statement = 'False'
                
                
                # Constructing the final text
                text = text = f"""
                You are an experienced journal editor, assess the future academic impact of the following research paper:
                - Title: {row['title']}
                - Abstract: {row['abstract']}
                - State-of-the-Art Performance: {'Yes' if SOTA == 1 else 'No'}
                - Released a New Dataset: {'Yes' if new_dataset else 'No'}
                - Code Available Open Access: {'Yes' if OA else 'No'}
                - Reference Quality Metric: {RQM} (on a scale from lowest 0 to highest 1)

                Based on the information provided, predict the future normalized academic impact of this paper, on a scale from lowest 0 to highest 1.
                """


                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                token_count = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum().item()
                if token_count > self.max_length:
                    print("The text has been truncated.")
      
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 14:
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                OA = row['OA']
                RQM = round(row['RQM'],1)

 
                dataset_statement = "This paper has released a new dataset." if new_dataset else "No new dataset has been released."

                oa_statement = 'This paper has open-accessed the code framework.' if OA else "The code is not available."
                if SOTA == 1:
                    sota_statement = 'True'
                else:
                    sota_statement = 'False'
                
                
                # Constructing the final text
                text = text = f"""
                You are an experienced journal editor, assess the future academic impact of the following research paper:
                - Title: {row['title']}
                - Abstract: {row['abstract']}
                - State-of-the-Art Performance: {'Yes' if SOTA == 1 else 'No'}

                Based on the information provided, predict the future normalized academic impact of this paper, on a scale from lowest 0 to highest 1.
                """


                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                token_count = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum().item()
                if token_count > self.max_length:
                    print("The text has been truncated.")
      
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 15:
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                OA = row['OA']
                RQM = round(row['RQM'],1)

 
                dataset_statement = "This paper has released a new dataset." if new_dataset else "No new dataset has been released."

                oa_statement = 'This paper has open-accessed the code framework.' if OA else "The code is not available."
                if SOTA == 1:
                    sota_statement = 'True'
                else:
                    sota_statement = 'False'
                
                
                # Constructing the final text
                text = text = f"""
                You are an experienced journal editor, assess the future academic impact of the following research paper:
                - Title: {row['title']}
                - Abstract: {row['abstract']}
                - Released a New Dataset: {'Yes' if new_dataset else 'No'}

                Based on the information provided, predict the future normalized academic impact of this paper, on a scale from lowest 0 to highest 1.
                """


                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                token_count = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum().item()
                if token_count > self.max_length:
                    print("The text has been truncated.")
      
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 16:
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                OA = row['OA']
                RQM = round(row['RQM'],1)

 
                dataset_statement = "This paper has released a new dataset." if new_dataset else "No new dataset has been released."

                oa_statement = 'This paper has open-accessed the code framework.' if OA else "The code is not available."
                if SOTA == 1:
                    sota_statement = 'True'
                else:
                    sota_statement = 'False'
                
                
                # Constructing the final text
                text = text = f"""
                You are an experienced journal editor, assess the future academic impact of the following research paper:
                - Title: {row['title']}
                - Abstract: {row['abstract']}
                - Code Available Open Access: {'Yes' if OA else 'No'}


                Based on the information provided, predict the future normalized academic impact of this paper, on a scale from lowest 0 to highest 1.
                """


                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                token_count = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum().item()
                if token_count > self.max_length:
                    print("The text has been truncated.")
      
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif self.prompt_style == 17:
                new_dataset = row['new_dataset']
                SOTA = row['SOTA']
                OA = row['OA']
                RQM = round(row['RQM'],1)

 
                dataset_statement = "This paper has released a new dataset." if new_dataset else "No new dataset has been released."

                oa_statement = 'This paper has open-accessed the code framework.' if OA else "The code is not available."
                if SOTA == 1:
                    sota_statement = 'True'
                else:
                    sota_statement = 'False'
                
                
                # Constructing the final text
                text = text = f"""
                You are an experienced journal editor, assess the future academic impact of the following research paper:
                - Title: {row['title']}
                - Abstract: {row['abstract']}
                - Reference Quality Metric: {RQM} (on a scale from lowest 0 to highest 1)

                Based on the information provided, predict the future normalized academic impact of this paper, on a scale from lowest 0 to highest 1.
                """


                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                token_count = (inputs['input_ids'] != self.tokenizer.pad_token_id).sum().item()
                if token_count > self.max_length:
                    print("The text has been truncated.")
      
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }     
def main(args):
    args.eff_gpus = int(torch.cuda.device_count())
    args.eff_batch_size = args.eff_gpus * args.batch_size
    
    if args.learning_rate is None:  # only base_lr is specified
        args.learning_rate = args.base_lr * args.eff_batch_size / 256
    
    
    # Load your dataset
    df = pd.read_csv(args.data_path)
    df_test = pd.read_csv(args.test_data_path)
        
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    

    device_map={'': torch.cuda.current_device()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, 
        num_labels=args.num_labels, 
        load_in_8bit=args.load_in_8bit,
        device_map=device_map,
    )

    

    model.config.pad_token_id = model.config.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model.loss_func = args.loss_func
    if len(args.target_modules)>0:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules.split(','),
            task_type=TaskType.SEQ_CLS,
            inference_mode=False
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            task_type=TaskType.SEQ_CLS,
            inference_mode=False
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        

    total_dataset = TextDataset(df, tokenizer, args.max_length,args.prompt_style)
    total_size = len(total_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size])
    test_dataset = TextDataset(df_test, tokenizer, args.max_length)
    # Prepare Accelerator
    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        default_tb_dir = datetime.now().strftime("%m-%d-%H-%M-%s")
        if args.runs_dir is None:
            args.runs_dir = os.path.join('ScImpactPredict/official_runs', default_tb_dir)
        os.makedirs(args.runs_dir,exist_ok=True)
        json_file_path = os.path.join(args.runs_dir,'args.json')
        save_args_to_json(args, json_file_path)
        
    # Define training arguments
    training_args = TrainingArguments(
        ddp_find_unused_parameters=False,
        output_dir=args.runs_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.total_epochs,
        logging_dir=args.runs_dir,
        logging_steps=10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        warmup_ratio=args.warmup_ratio,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train model
    model, tokenizer = accelerator.prepare(model, tokenizer)
    trainer.train()
    
    if accelerator.is_local_main_process:
        model_last_id = os.path.join(args.runs_dir, 'last')
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            model_last_id,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        score_state_dict = unwrapped_model.score.state_dict()
        print(score_state_dict)
        torch.save(score_state_dict, os.path.join(model_last_id, 'score.pt'))
def save_args_to_json(args, file_path):
     
    args_dict = vars(args)
     
    with open(file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
def get_args():
    parser = argparse.ArgumentParser(description="Train a transformer model with LoRA adaptation on text classification tasks.")
    
    # Dataset and training configuration
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum length of the tokenized input sequences')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--total_epochs', type=int, default=5, help='Total number of epochs to train')
    parser.add_argument('--base_lr', type=float, default=5e-5, help='Base learning rate for the optimizer')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for the optimizer')
    parser.add_argument('--data_path', type=str, default='ScImpactPredict/NAID/NAID_train.csv', help='Path to the training dataset CSV file')
    parser.add_argument('--test_data_path', type=str, default='ScImpactPredict/NAID/NAID_test.csv', help='Path to the testing dataset CSV file')
    parser.add_argument('--checkpoint', type=str, default='llama3_weight', help='Model checkpoint path')
    parser.add_argument('--loss_func', type=str, default='mse', choices=['bce', 'mse', 'l1','smoothl1','focalmse'], help='Loss function to use')
    parser.add_argument('--num_labels', type=int, default=1, help='Number of labels for sequence classification')
    parser.add_argument('--load_in_8bit', type=bool, default=True, help='Whether to load the model in 8-bit for efficiency')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train the model on (cuda or cpu)')
    parser.add_argument('--lora_r', type=int, default=16, help='Rank of LoRA layers')
    parser.add_argument('--lora_alpha', type=int, default=32, help='Expansion factor for LoRA layers')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='Dropout rate for LoRA layers')
    parser.add_argument('--target_modules', type=str, default='q_proj,v_proj', help='Comma-separated list of transformer modules to apply LoRA')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for learning rate scheduler')
    
    parser.add_argument('--prompt_style', type=int, default=0, )
    parser.add_argument('--data_augmentation', type=str, default='none', choices=['none', 'synonym'], help='Data augmentation technique to use')



    
    
    parser.add_argument('--runs_dir', type=str, default=None, help='Directory for storing TensorBoard logs and model checkpoints')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
