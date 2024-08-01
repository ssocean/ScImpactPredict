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
 
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    parser.add_argument('--data_style', type=int,default=1)
    parser.add_argument('--test_ratio', type=float,default=1.0)
    parser.add_argument('--threshold', type=float,default=0.5)
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

args = get_args()
args.eff_gpus = int(torch.cuda.device_count() * args.batch_size)

writer = SummaryWriter(args.runs_dir)


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = float(row['status'])
        if args.data_style == 0:

            text = f"Given a certain paper, Title: {row['title'].strip()}\n Abstract: {row['abstract'].strip()}. \n Predict its normalized scholar impact (between 0 and 1):"
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 88:
            text = f'''Given a certain paper, Title: {row['title'].strip()}\n Abstract: {row['abstract'].strip()}. \n Predict its normalized academic impact (between 0 and 1):'''
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
tokenizer.pad_token = tokenizer.eos_token
device_map={'':torch.cuda.current_device()}

model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, num_labels=args.num_labels, load_in_8bit=args.load_in_8bit,device_map=device_map,)
model.config.pad_token_id = model.config.eos_token_id
model.loss_func = args.loss_func
model = PeftModelForTokenClassification.from_pretrained(model, args.weight_dir)# 不改变score 权重
# model.score.load_state_dict(torch.load(os.path.join(args.weight_dir,'last_score.pt')))
model = model.merge_and_unload()
print(model.score.weight)
full_data = pd.read_csv(args.data_path)
dataset = TextDataset(full_data, tokenizer, max_length=args.max_length)
# dataset = TextDataset(full_data, tokenizer)


test_size = int(args.test_ratio * len(dataset)) 
train_size = len(dataset) -test_size


test_dataset,train_dataset = random_split(dataset, [ test_size,train_size])


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f'Test Dataloader has {len(test_loader)} samples in total')


# Define variables to store True Positives, False Positives, and False Negatives
TP = 0
FP = 0
FN = 0
with torch.no_grad():
    for batch in tqdm(test_loader):
        
        outputs = model(**batch)
        predictions = outputs["logits"].squeeze(1)  # Assuming logits are the direct outputs you're interested in
        labels = batch["labels"]

        # Convert predictions to binary outcomes (0 or 1)
        predicted_positives = (predictions > args.threshold).int()  # Convert boolean mask to integer

        # Ensure labels are also integers if they are not already
        labels = labels.int()

        # Calculate True Positives, False Positives, and False Negatives
        TP += (predicted_positives & labels).sum().item()
        FP += (predicted_positives & (~labels.bool())).sum().item()  # Convert labels to boolean for bitwise operation, then to int
        FN += ((~predicted_positives.bool()) & labels).sum().item()  # Convert predicted_positives to boolean for bitwise operation

# Calculate Precision, Recall, and F1 Score
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

writer.close()

