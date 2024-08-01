from datetime import datetime
import logging
import random
import time
import torch
from torch.nn.functional import mse_loss,l1_loss
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
from accelerate import Accelerator
import os
import torch.nn as nn

from tools.order_metrics import NDCG_20, NDCG_k
LlamaForSequenceClassification
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import transformers.models.qwen2
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
accelerator = Accelerator()
def init_logger(out_pth: str = 'logs'):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    os.makedirs(out_pth,exist_ok=True)
    handler = logging.FileHandler(fr'{out_pth}/{time.strftime("%Y_%b_%d", time.localtime())}_log.txt')

    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s # %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger
def set_seed(seed):
    if seed>=0:
        # Python random seed
        random.seed(seed)
        
        # Numpy random seed
        np.random.seed(seed)
        
        # PyTorch random seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        
        # CuDNN deterministic setting
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set the seed

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
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--total_epochs', type=int, default=3, help='Total number of epochs to train')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate for the optimizer')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for learning rate scheduler')
    parser.add_argument('--data_path', type=str, default='/home/u1120220285/ScitePredict/NAID/NAID_train.csv', help='Path to the dataset CSV file')
    parser.add_argument('--test_data_path', type=str, default=None, help='Path to the dataset CSV file')
    parser.add_argument('--val_data_path', type=str, default=None, help='Path to the dataset CSV file')
    parser.add_argument('--checkpoint', type=str, default='/home/u1120220285/llama3_weight', help='Model checkpoint path')
    parser.add_argument('--loss_func', type=str, default='mse',choices=['bce','mse','l1'])
    parser.add_argument('--data_style', type=int,default=1)
    parser.add_argument('--weight_decay', type=float,default=1e-4)
    parser.add_argument('--train_ratio', type=float, default=0.9, help='Device to train the model on (cuda or cpu)')
    # parser.add_argument('--model_save_path', type=str, help='Path to save the trained models')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train the model on (cuda or cpu)')
        # Training configuration
    parser.add_argument('--save_internal', type=int,default=1)
    parser.add_argument('--seed', type=int,default=-1)
    parser.add_argument('--num_labels', type=int, default=1, help='Number of labels for sequence classification')
    parser.add_argument('--load_in_8bit', type=bool, default=True, help='Whether to load the model in 8-bit for efficiency')
    parser.add_argument('--max_norm', type=float, default=1.0, help='Clip grad max norm')
    
    parser.add_argument('--exchange_test_val', type=bool, default=False, help='Whether to load the model in 8-bit for efficiency')
    # LoRA configuration
    parser.add_argument('--lora_r', type=int, default=16, help='Rank of LoRA layers')
    parser.add_argument('--lora_alpha', type=int, default=32, help='Expansion factor for LoRA layers')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='Dropout rate for LoRA layers')
    parser.add_argument('--lora_bias', type=str, default='none', help='Bias mode for LoRA layers')
    parser.add_argument('--target_modules', type=str, default='q_proj,v_proj', help='Comma-separated list of transformer modules to apply LoRA')
    
    
    
    # 默认 TensorBoard 日志目录使用当前日期和时间
    default_tb_dir = datetime.now().strftime("%m-%d-%H-%M")
    # 可以参考os.path.join('/home/u1120220285/xxx/ScitePredict/runs',default_tb_dir)
    parser.add_argument('--runs_dir', type=str, default=None, help='Directory for storing TensorBoard logs')

    return parser.parse_args()


args = get_args()
args.eff_gpus = int(torch.cuda.device_count())
args.eff_batch_size = args.eff_gpus * args.batch_size
set_seed(args.seed)

if args.learning_rate is None:  # only base_lr is specified
    args.learning_rate = args.base_lr * args.eff_batch_size / 256
    
if accelerator.is_local_main_process:
    writer = SummaryWriter(log_dir=args.runs_dir)

    args.runs_dir =  writer.log_dir
    args.logs_dir = os.path.join(args.runs_dir,'logs')
    logger = init_logger(args.logs_dir)

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = float(row['TNCSI'])
        if args.data_style == 0:

            text = f"Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}. \n Predict its normalized scholar impact (between 0 and 1):"
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 1:    
            most_cite = int(row['authors_title'])
            oa = row['OA']
            if oa is None or oa == 'None':
                oa = '\n The code of the paper has been shared on the web. \n'
            else:
                oa = '\n The code of the paper is unavailable. \n'
            if most_cite >10000:
                authors_title = '\n This article was written by a renowned scholar in the field. \n'
            else:
                authors_title = '\n The author of this article is not very well-known in their field. \n'
            text = f"Given a certain paper entitled {row['title']}, predict its normalized scholar impact (between 0 and 1).{oa}{authors_title} The Abstract of paper: {row['abstract']}."
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 6:
            text = f'''Title: {row['title']}\n Abstract: {row['abstract']}'''
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 7:
            text = f'''Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}. \n Predict its academic impact:'''
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 8:
            text = f'''Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}. \n Predict its normalized academic impact (between 0 and 1):'''
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 9:
            text = f'''Given the provided title and abstract, predict the future normalized academic impact on a scale from 0 (lowest impact) to 1 (highest impact). You may consider factors such as the language clarity, novelty of the research, or the claim of state-of-the-art, etc. Title: {row['title']}\n Abstract: {row['abstract']}'''
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        else:
            raise Exception('data style not supported')
            


# Setup device and model configuration
# device = torch.device(args.device if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
tokenizer.pad_token = tokenizer.eos_token
device_map={'':torch.cuda.current_device()}
model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint, num_labels=args.num_labels, load_in_8bit=args.load_in_8bit,device_map=device_map,)

model.config.pad_token_id = model.config.eos_token_id
model.loss_func = args.loss_func


 
# Define LoRA Config
# Setup LoRA
lora_config = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    bias=args.lora_bias,
    inference_mode=False,
    target_modules=args.target_modules.split(','),
    task_type=TaskType.SEQ_CLS 
)

# prepare int-8 model for training
model = prepare_model_for_kbit_training(model)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()




total_epoch = args.total_epochs
if args.weight_decay >= 0:
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
else:
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)



# LR adjust


if args.test_data_path:
    test_data = pd.read_csv(args.test_data_path)
    test_dataset = TextDataset(test_data, tokenizer, max_length=args.max_length)
    if args.val_data_path:
        train_data = pd.read_csv(args.data_path)
        train_dataset = TextDataset(train_data, tokenizer, max_length=args.max_length)    

        val_data = pd.read_csv(args.val_data_path)
        val_dataset = TextDataset(val_data, tokenizer, max_length=args.max_length)    
    else:
        
        full_data = pd.read_csv(args.data_path)
        dataset = TextDataset(full_data, tokenizer, max_length=args.max_length)    
        train_size = int(args.train_ratio * len(dataset)) 
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    print(f'Train Samples:{len(train_dataset)}, Val Samples:{len(val_dataset)}')
else:
    args.train_ratio = 0.8
    full_data = pd.read_csv(args.data_path)
    dataset = TextDataset(full_data, tokenizer, max_length=args.max_length)
    # dataset = TextDataset(full_data, tokenizer)

    train_size = int(args.train_ratio * len(dataset)) 
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size
    print(f'Train Samples:{train_size }, Val Samples:{val_size}, Test Samples:{test_size}')


    train_dataset, val_dataset,test_dataset = random_split(dataset, [train_size, val_size,test_size])

    # Extract indices of the test_dataset for saving it to CSV
    

    # Save test dataset to CSV
            # 合并训练数据和验证数据的索引
        # combined_train_indices = train_indices + val_indices
        # train_data = full_data.iloc[combined_train_indices]

        # # 存储训练数据到指定文件夹
        # train_data.to_csv(os.path.join(args.runs_dir, 'train_data.csv'), index=False)
    if accelerator.is_local_main_process:
        
        train_indices = train_dataset.indices
        train_data  = full_data.iloc[train_indices]
        train_data.to_csv(os.path.join(args.runs_dir , 'train_data.csv'), index=False)
        
        val_indices = val_dataset.indices
        val_data  = full_data.iloc[val_indices]
        val_data.to_csv(os.path.join(args.runs_dir , 'val_data.csv'), index=False)

        test_indices = test_dataset.indices
        test_data = full_data.iloc[test_indices]
        test_data.to_csv(os.path.join(args.runs_dir , 'test_data.csv'), index=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

print(f'device {str(accelerator.device)} is used!')

total_steps = len(train_dataset) / accelerator.num_processes *total_epoch
warmup_steps =  args.warmup_ratio * total_steps


def lr_lambda(current_step: int):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(
        0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
    )
scheduler = LambdaLR(optimizer, lr_lambda)

if args.exchange_test_val:
    temp = test_loader 
    test_loader = val_loader
    val_loader = temp

train_loader, val_loader, test_loader, model, optimizer, scheduler = accelerator.prepare(train_loader, val_loader,test_loader, model, optimizer, scheduler)



best_val_mae = float('inf')  # 初始化最佳验证损失

if accelerator.is_local_main_process:
    main_process_step = 0
    json_file_path = os.path.join(args.runs_dir,'args.json')
    save_args_to_json(args, json_file_path)

for epoch in range(1,total_epoch+1):  # Number of epochs
    model.train()
    train_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=(not accelerator.is_local_main_process))
    for  i,batch in pbar:
        # print(f'train batch:{batch}')
        # batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        # print(model.roberta.embeddings.word_embeddings.weight)
        # print(f'train sample: Pred[0]: {outputs["logits"][0].item()} ----- GT[0]: {batch["labels"][0].item()} ---- AVG Loss:{loss.item()} ------ lr: {optimizer.param_groups[0]["lr"]}' )
        # loss.backward()
        accelerator.backward(loss)
        if args.max_norm>0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
        
        optimizer.step()
        pbar.set_description(f"Train {epoch} iter {i}: {args.loss_func} loss {loss.item():.4f}. lr {scheduler.get_last_lr()[0]:e}")
        
        scheduler.step()
        optimizer.zero_grad()
        train_loss += loss.item()

        if accelerator.is_local_main_process:
            main_process_step+=1
            writer.add_scalar(f"Train {args.loss_func} Loss per Batch (Main GPU)",  loss.item(), main_process_step)  # 将平均MSE写入TensorBoard
            

    train_loss = torch.Tensor([train_loss]).to(accelerator.device)
    
    train_loss = accelerator.gather(train_loss).sum().item() 
    avg_train_loss = train_loss / len(train_loader) / args.eff_gpus 
    if accelerator.is_local_main_process:
        writer.add_scalar(f"Train AVG {args.loss_func} LOSS per Epoch (ALL GPU)",  avg_train_loss, epoch)  # 将平均MSE写入TensorBoard
    
    
    model.eval()
    total_val_mse = 0.0
    total_model_loss = 0.0
    total_val_mae = 0.0
    with torch.no_grad():
        vbar = tqdm(enumerate(val_loader), total=len(val_loader), disable=(not accelerator.is_local_main_process))
        for i,batch in vbar:
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = outputs["logits"]
            labels = batch["labels"]
            # Calculate MSE loss for the current batch
            if args.loss_func == 'bce':
                val_mse = mse_loss(accelerator.gather(nn.Sigmoid()(predictions.squeeze())), accelerator.gather(labels))
                val_mse = l1_loss(accelerator.gather(nn.Sigmoid()(predictions.squeeze())), accelerator.gather(labels))
                # accelerator.print(nn.Sigmoid()(predictions.squeeze()),labels)
                accelerator.print(f'val_mse: {val_mse}')
                vbar.set_description(f"Val {epoch} iter {i}: mse loss {val_mse:.4f}, mae loss {val_mae:.4f}.")
            else:
                val_mse = mse_loss(accelerator.gather(predictions.squeeze()), accelerator.gather(labels))
                val_mae =l1_loss(accelerator.gather(predictions.squeeze()), accelerator.gather(labels))
                # accelerator.print(predictions.squeeze(),labels)
                # if accelerator.is_local_main_process:
                accelerator.print(predictions.squeeze(),labels)
                vbar.set_description(f"Val {epoch} iter {i}: mse loss {val_mse:.4f}, mae loss {val_mae:.4f}.")
                

            
            device_sum_val_mse = accelerator.gather(val_mse).sum()
            device_sum_val_mae = accelerator.gather(val_mae).sum()
            
            total_val_mse += device_sum_val_mse.item()
            total_val_mae += device_sum_val_mae.item()
            


        avg_val_mse = total_val_mse / len(val_loader) / args.eff_gpus 
        avg_val_mae = total_val_mae / len(val_loader) / args.eff_gpus 

        
        # avg_mse_loss = total_val_mse / len(val_loader)  # 计算所有批次的平均MSE
        # total_model_loss = total_model_loss/ len(val_loader)
        
        # accelerator.print(f'Epoch {epoch} val {args.loss_func}: {total_model_loss}; MSE: {avg_mse_loss} ')
        if accelerator.is_local_main_process:
            writer.add_scalar(f"Val AVG MSE LOSS per Epoch (ALL GPU)", avg_val_mse, epoch)  # 将平均MSE写入TensorBoard
            writer.add_scalar("Val AVG MAE LOSS per Epoch (ALL GPU)", avg_val_mae, epoch)  # 将平均MSE写入TensorBoard
            
    total_test_mse = 0.0
    total_test_mae = 0.0
    model.eval()
    all_pred = []
    all_GT = []

    with torch.no_grad():
        tbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for i,batch in tbar:
            outputs = model(**batch)
            predictions = outputs["logits"]
            labels = batch["labels"].cuda()
            all_GT+=labels
            all_pred+= predictions.squeeze(1)
            test_mse = mse_loss(accelerator.gather(predictions.squeeze(1)), accelerator.gather(labels))
            test_mae = l1_loss(accelerator.gather(predictions.squeeze(1)), accelerator.gather(labels))
            
            tbar.set_description(f"Test iter {i}: mse loss {test_mse:.4f}, mae loss {test_mae:.4f}.")
            accelerator.print(predictions.squeeze(),labels)
            device_sum_test_mse = accelerator.gather(test_mse).sum()
            device_sum_test_mae = accelerator.gather(test_mae).sum()
            
            total_test_mse += device_sum_test_mse.item()
            total_test_mae += device_sum_test_mae.item()


        avg_test_mse = total_test_mse / len(test_loader) / args.eff_gpus 
        avg_test_mae = total_test_mae / len(test_loader) / args.eff_gpus 
        all_pred = accelerator.gather(all_pred)
        all_GT= accelerator.gather(all_GT)
        all_pred = torch.stack([p for p in all_pred]).squeeze().reshape(-1)
        all_GT = torch.stack([g for g in all_GT]).squeeze().reshape(-1)
        if accelerator.is_local_main_process:
            writer.add_scalar(f"Test AVG MSE LOSS per epoch",avg_test_mse, epoch)  # 将平均MSE写入TensorBoard
            writer.add_scalar("Test AVG MAE LOSS per epoch", avg_test_mae, epoch)  # 将平均MSE写入TensorBoard
        
        if accelerator.is_local_main_process:
            logger.info(f'test_avg_mse:{test_mse}, test_avg_mae:{test_mae}')
            logger.info(f'NDCG:{NDCG_k(all_pred,all_GT)}')    
    accelerator.wait_for_everyone()
    
    if accelerator.is_local_main_process:
        model_last_id = os.path.join(args.runs_dir,'last')
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
        model_last_id,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
            )
        score_state_dict = unwrapped_model.score.state_dict()
        print(score_state_dict)
        torch.save(score_state_dict, os.path.join(model_last_id,f'score.pt'))

        if epoch % args.save_internal == 0 :
            model_save_id = os.path.join(args.runs_dir,f'model_{epoch}ep')
            best_val_mae = total_val_mae
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
            model_save_id,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
                )

            score_state_dict = unwrapped_model.score.state_dict()
            
            

            torch.save(score_state_dict, os.path.join(model_save_id,f'score.pt'))

        if total_val_mae < best_val_mae:
            model_save_id = os.path.join(args.runs_dir,'best')
            best_val_mae = total_val_mae
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
            model_save_id,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
                )
            score_state_dict = unwrapped_model.score.state_dict()
            
            
  
            torch.save(score_state_dict, os.path.join(model_save_id,f'score.pt'))
        logger.info(f'Epoch {epoch}---Train MSE: {avg_train_loss}, Val MSE: {avg_val_mse}, Val MAE: {avg_val_mae}')
        




total_test_mse = 0.0
total_test_mae = 0.0
model.eval()
all_pred = []
all_GT = []

with torch.no_grad():
    
    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    for i,batch in pbar:
        outputs = model(**batch)
        predictions = outputs["logits"]
        labels = batch["labels"].cuda()
        all_GT+=labels
        all_pred+= predictions.squeeze(1)
        test_mse = mse_loss(accelerator.gather(predictions.squeeze(1)), accelerator.gather(labels))
        test_mae = l1_loss(accelerator.gather(predictions.squeeze(1)), accelerator.gather(labels))
        
        pbar.set_description(f"Test iter {i}: mse loss {test_mse:.4f}, mae loss {test_mae:.4f}.")
        accelerator.print(predictions.squeeze(),labels)
        device_sum_test_mse = accelerator.gather(test_mse).sum()
        device_sum_test_mae = accelerator.gather(test_mae).sum()
        
        total_test_mse += device_sum_test_mse.item()
        total_test_mae += device_sum_test_mae.item()
        if accelerator.is_local_main_process:
            writer.add_scalar(f"Test MSE LOSS per Batch",test_mse/args.eff_gpus, i)  # 将平均MSE写入TensorBoard
            writer.add_scalar("Test MAE LOSS per Batch", test_mse/args.eff_gpus, i)  # 将平均MSE写入TensorBoard

    avg_test_mse = total_test_mse / len(test_loader) / args.eff_gpus 
    avg_test_mae = total_test_mae / len(test_loader) / args.eff_gpus 
    all_pred = accelerator.gather(all_pred)
    all_GT= accelerator.gather(all_GT)
    all_pred = torch.stack([p for p in all_pred]).squeeze().reshape(-1)
    all_GT = torch.stack([g for g in all_GT]).squeeze().reshape(-1)

    
    if accelerator.is_local_main_process:
        logger.info(f'test_avg_mse:{test_mse}, test_avg_mae:{test_mae}')
        logger.info(f'NDCG:{NDCG_20(all_pred,all_GT)}')

 
if accelerator.is_local_main_process:
    writer.close()
