from datetime import datetime
import random
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
from sklearn.metrics import ndcg_score
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 固定随机种子
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 设置 cuDNN 确定性和禁用自动优化
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def NDCG_k(predictions, labels, k=20):
    if len(predictions) < k:
        return -1  # or handle as preferred
    return ndcg_score([labels], [predictions], k=k)
def save_args_to_json(args, file_path):
    # 将 args 对象转换为字典
    args_dict = vars(args)
    # 将字典保存到 JSON 文件
    with open(file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
def get_args():
    parser = argparse.ArgumentParser(description="Train a transformer model with LoRA adaptation on text classification tasks.")
    
    # Dataset and training configuration
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum length of the tokenized input sequences')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training and validation')


    parser.add_argument('--data_path', type=str, default='/home/u1120220285/ScitePredict/data/Data_TNCSI_S_OA_AuthorCite_8242_fix1.csv', help='Path to the dataset CSV file')
    parser.add_argument('--checkpoint', type=str, default='/home/u1120220285/llama3_weight', help='Model checkpoint path')
    parser.add_argument('--weight_dir', type=str, default='runs/Jul12_14-54-26_gpu22', help='Model checkpoint path')
    parser.add_argument('--loss_func', type=str, default='bce',choices=['bce','mse','l1'])
    parser.add_argument('--prompt_style', type=int,default=0)
    parser.add_argument('--test_ratio', type=float,default=1.0)
    # parser.add_argument('--model_save_path', type=str, help='Path to save the trained models')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train the model on (cuda or cpu)')


    parser.add_argument('--num_labels', type=int, default=1, help='Number of labels for sequence classification')
    parser.add_argument('--load_in_8bit', type=bool, default=True, help='Whether to load the model in 8-bit for efficiency')
    
    # LoRA configuration
    parser.add_argument('--lora_r', type=int, default=16, help='Rank of LoRA layers')
    parser.add_argument('--lora_alpha', type=int, default=32, help='Expansion factor for LoRA layers')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='Dropout rate for LoRA layers')
    parser.add_argument('--lora_bias', type=str, default='none', help='Bias mode for LoRA layers')
    parser.add_argument('--target_modules', type=str, default='q_proj,v_proj', help='Comma-separated list of transformer modules to apply LoRA')
    
    
    
    # 默认 TensorBoard 日志目录使用当前日期和时间
    default_tb_dir = datetime.now().strftime("%m-%d-%H-%M")
    parser.add_argument('--runs_dir', type=str, default=os.path.join('/home/u1120220285/ScitePredict/inference',default_tb_dir), help='Directory for storing TensorBoard logs')

    return parser.parse_args()
from accelerate import Accelerator
accelerator = Accelerator()
args = get_args()
class TextDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=1024):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            label = float(row['TNCSI_SP'])
            if args.prompt_style == 1:
                text = f'''Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}. \n Predict its normalized academic impact:'''
                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif args.prompt_style == 0:
                text = f'''Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}. \n Predict its normalized academic impact (between 0 and 1):'''
                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif args.prompt_style == 2:
                text = f'''Given the provided title and abstract, predict the future normalized academic impact on a scale from 0 (lowest impact) to 1 (highest impact). You may consider factors such as the language clarity, novelty of the research, or the claim of state-of-the-art, etc. Title: {row['title']}\n Abstract: {row['abstract']}'''
                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif args.prompt_style == 3:
                text = f'''Title: {row['title']}\n Abstract: {row['abstract']}'''
                inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                        return_tensors="pt")
                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'labels': torch.tensor(label, dtype=torch.float)
                }
            elif args.prompt_style == 11:
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
            elif args.prompt_style == 12:
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
            elif args.prompt_style == 13:
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
            elif args.prompt_style == 14:
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
            elif args.prompt_style == 15:
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
            elif args.prompt_style == 16:
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
            elif args.prompt_style == 17:
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
args.eff_gpus = int(torch.cuda.device_count() * args.batch_size)

writer = SummaryWriter(args.runs_dir)


tokenizer = AutoTokenizer.from_pretrained(args.weight_dir)
tokenizer.pad_token = tokenizer.eos_token
device_map={'':torch.cuda.current_device()}

# model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, num_labels=args.num_labels, load_in_8bit=args.load_in_8bit,device_map=device_map,)
from peft import AutoPeftModelForCausalLM,AutoPeftModelForSequenceClassification,AutoPeftModelForTokenClassification
# model = PeftModelForTokenClassification.from_pretrained(model, os.path.join(args.weight_dir,'last'))# 不改变score 权重
model = AutoPeftModelForSequenceClassification.from_pretrained(args.weight_dir, num_labels=args.num_labels, load_in_8bit=args.load_in_8bit,device_map=device_map,)# 不改变score 权重
model.config.pad_token_id = model.config.eos_token_id
model.loss_func = args.loss_func
# model.score.load_state_dict(torch.load(os.path.join(args.weight_dir,'score.pt')))
# model = model.merge_and_unload()
print(model.score.weight)

full_data = pd.read_csv(args.data_path)
dataset = TextDataset(full_data, tokenizer, max_length=args.max_length)
test_loader = DataLoader(dataset, batch_size=16, shuffle=True)
print(f'Test Dataloader has {len(test_loader)} samples in total')
total_val_mse = 0.0
total_val_mae = 0.0

all_pred = []
all_GT = []
model.eval()
model,test_loader = accelerator.prepare(model,test_loader)
with torch.no_grad():
    for batch in tqdm(test_loader):

        outputs = model(**batch)
        predictions = outputs["logits"]
        labels = batch["labels"]
        all_GT+=labels

        all_pred+= predictions.squeeze(1)

        mse = nn.MSELoss()(predictions.squeeze(1), labels).item()
        mae = nn.L1Loss()(predictions.squeeze(1), labels).item()

        total_val_mse += mse
        total_val_mae += mae
        


    avg_mse_loss = total_val_mse / len(test_loader)
    avg_mae_loss = total_val_mae / len(test_loader)# 计算所有批次的平均MSE


all_pred = torch.Tensor(all_pred).squeeze()
all_GT = torch.Tensor(all_GT).squeeze()
# print(all_pred.shape)
# print(all_GT.shape)
# print(NDCG_20(all_pred,all_GT))
print(args.weight_dir)
print(avg_mse_loss,avg_mae_loss)
print(NDCG_k(all_pred,all_GT))
writer.close()

