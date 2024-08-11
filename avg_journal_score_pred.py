from datetime import datetime
import random
import numpy as np
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, AdamW, RobertaForSequenceClassification, AutoTokenizer, \
    AutoModelForSequenceClassification, FlaxLlamaForCausalLM, LlamaForSequenceClassification
import pandas as pd
from torch.utils.data.dataset import random_split
from tqdm import tqdm
import os
import argparse
import json
import torch
from peft import PeftModel,PeftModelForTokenClassification
import os
import torch.nn as nn

import transformers.models.qwen2
from peft import AutoPeftModelForCausalLM,AutoPeftModelForSequenceClassification,AutoPeftModelForTokenClassification
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def save_args_to_json(args, file_path):

    args_dict = vars(args)

    with open(file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
def get_args():
    parser = argparse.ArgumentParser(description="Train a transformer model with LoRA adaptation on text classification tasks.")
    
    # Dataset and training configuration
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum length of the tokenized input sequences')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training and validation')


    parser.add_argument('--data_path', type=str, default='ScImpactPredict/data/Data_TNCSI_S_OA_AuthorCite_8242_fix1.csv', help='Path to the dataset CSV file')
    parser.add_argument('--checkpoint', type=str, default='llama3_weight', help='Model checkpoint path')
    parser.add_argument('--weight_dir', type=str, default='runs/Jul12_14-54-26_gpu22', help='Model checkpoint path')
    parser.add_argument('--loss_func', type=str, default='bce',choices=['bce','mse','l1'])
    parser.add_argument('--prompt_style', type=int,default=0)
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
    
    default_tb_dir = datetime.now().strftime("%m-%d-%H-%M")
    parser.add_argument('--runs_dir', type=str, default=os.path.join('ScImpactPredict/inference',default_tb_dir), help='Directory for storing TensorBoard logs')

    return parser.parse_args()

args = get_args()
args.eff_gpus = int(torch.cuda.device_count() * args.batch_size)




class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = 0
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
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
tokenizer.pad_token = tokenizer.eos_token
device_map={'':torch.cuda.current_device()}


model = AutoPeftModelForSequenceClassification.from_pretrained(args.weight_dir, num_labels=args.num_labels, load_in_8bit=args.load_in_8bit,device_map=device_map,) 
model.config.pad_token_id = model.config.eos_token_id
model.loss_func = args.loss_func

print(model.score.weight)
full_data = pd.read_csv(args.data_path)
dataset = TextDataset(full_data, tokenizer, max_length=args.max_length)
test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

print(f'Test Dataloader has {len(test_loader)} samples in total')


import torch
from tqdm import tqdm

all_predictions = []
model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader):
        outputs = model(**batch)
        predictions = outputs["logits"].squeeze(1)
        all_predictions.append(predictions)


all_predictions = torch.cat(all_predictions, dim=0)


average_prediction = all_predictions.mean()


sorted_predictions = torch.sort(all_predictions)
top_5_num = int(len(dataset)*0.01)
top_10_num = int(len(dataset)*0.1)
top_25_num = int(len(dataset)*0.25)

top_5_predictions = sorted_predictions.values[-top_5_num:]
top_10_predictions = sorted_predictions.values[-top_10_num:]

top_25_predictions = sorted_predictions.values[-top_25_num:]
average_top_5 = top_5_predictions.mean()
average_top_10 = top_10_predictions.mean()
average_top_25 = top_25_predictions.mean()

print("Average prediction:", average_prediction.item())
print("Average of top 5% predictions:", average_top_5.item())
print("Average of top 10% predictions:", average_top_10.item())
print("Average of top 25% predictions:", average_top_25.item())


