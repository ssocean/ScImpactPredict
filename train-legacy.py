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
from accelerate import Accelerator
import os
import torch.nn as nn
LlamaForSequenceClassification

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import transformers.models.qwen2
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
accelerator = Accelerator()
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
    parser.add_argument('--total_epochs', type=int, default=10, help='Total number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate for the optimizer')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for learning rate scheduler')
    parser.add_argument('--data_path', type=str, default='/home/u1120220285/ScitePredict/tools/Data_TNCSI_S_OA_AuthorCite_8242_fix1.csv', help='Path to the dataset CSV file')
    parser.add_argument('--checkpoint', type=str, default='/home/u1120220285/llama3_weight', help='Model checkpoint path')
    parser.add_argument('--loss_func', type=str, default='mse',choices=['bce','mse','l1'])
    parser.add_argument('--data_style', type=int,default=1)
    
    # parser.add_argument('--model_save_path', type=str, help='Path to save the trained models')
    parser.add_argument('--device', type=str, default='cuda', help='Device to train the model on (cuda or cpu)')
        # Training configuration

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
    # 可以参考os.path.join('/home/u1120220285/xxx/ScitePredict/runs',default_tb_dir)
    parser.add_argument('--runs_dir', type=str, default=None, help='Directory for storing TensorBoard logs')

    return parser.parse_args()

args = get_args()
args.eff_gpus = int(torch.cuda.device_count() * args.batch_size)

if accelerator.is_local_main_process:
    # os.path.join('/home/u1120220285/ScitePredict/runs',default_tb_dir),
    
    writer = SummaryWriter(log_dir=args.runs_dir)

    args.runs_dir =  writer.log_dir
    print('-'*50)
    print(f'Log has been saved to {writer.log_dir}')

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

            text = f"Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}. \n Predict its normalized scholar impact:"
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 2:

            text = f"Title: {row['title']}\n Abstract: {row['abstract']}."
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 3:

            text = f"Given the following details of a research paper, Title: {row['title']}\n Abstract: {row['abstract']}. \n Predict its normalized impact factor of this paper on a scale from 0 to 1:"
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
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
    target_modules=args.target_modules.split(','),
    task_type=TaskType.TOKEN_CLS
)

# prepare int-8 model for training
model = prepare_model_for_kbit_training(model)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()


full_data = pd.read_csv(args.data_path)
dataset = TextDataset(full_data, tokenizer, max_length=args.max_length)
# dataset = TextDataset(full_data, tokenizer)

train_size = int(0.8 * len(dataset)) 
val_size = len(dataset) - train_size



train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)






# 假设 model 是你的预训练模型
total_epoch = args.total_epochs
# args.model_save_path = os.path.join(args.runs_dir,)f'weights/{args.checkpoint.split("/")[-1]}-{total_epoch}ep-balanced2'  # 模型保存的路径
# 创建优化器
optimizer = AdamW(model.parameters(), lr=args.learning_rate)



# 定义总训练步数和预热步数
total_steps = len(train_dataset) / accelerator.num_processes *total_epoch
warmup_steps =  args.warmup_ratio * total_steps

# 定义学习率调整函数
def lr_lambda(current_step: int):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(
        0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
    )

# 创建 LambdaLR 调度器
scheduler = LambdaLR(optimizer, lr_lambda)



print(f'device {str(accelerator.device)} is used!')
train_loader, val_loader, model, optimizer, scheduler = accelerator.prepare(train_loader, val_loader, model, optimizer, scheduler)



best_bce_loss = float('inf')  # 初始化最佳验证损失

if accelerator.is_local_main_process:
    main_process_step = 0
    json_file_path = os.path.join(args.runs_dir,'args.json')
    save_args_to_json(args, json_file_path)

for epoch in range(total_epoch):  # Number of epochs
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
        optimizer.step()
        pbar.set_description(f"epoch {epoch + 1} iter {i}: {args.loss_func} loss {loss.item():.4f}. lr {scheduler.get_last_lr()[0]:e}")
        
        scheduler.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        # 更新学习率
        if accelerator.is_local_main_process:
            main_process_step+=1
            writer.add_scalar(f"{args.loss_func} loss per Step",  loss.item(), main_process_step)  # 将平均MSE写入TensorBoard
            

    train_loss = torch.Tensor([train_loss]).to(accelerator.device)
    
    train_loss = accelerator.gather(train_loss).sum().item() 
    avg_train_loss = train_loss / len(train_loader) / int(torch.cuda.device_count())
    if accelerator.is_local_main_process:
        writer.add_scalar(f"Train AVG {args.loss_func}  LOSS per Epoch",  avg_train_loss, epoch)  # 将平均MSE写入TensorBoard
    
    
    model.eval()
    total_val_mse = 0.0
    total_model_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = outputs["logits"]
            labels = batch["labels"]
            # Calculate MSE loss for the current batch
            if args.loss_func == 'bce':
                val_mse = mse_loss(accelerator.gather(nn.Sigmoid()(predictions.squeeze())), accelerator.gather(labels))
                accelerator.print(nn.Sigmoid()(predictions.squeeze()),labels)
                accelerator.print(f'val_mse: {val_mse}')
            else:
                val_mse = mse_loss(accelerator.gather(predictions.squeeze()), accelerator.gather(labels))
                accelerator.print(predictions.squeeze(),labels)
                accelerator.print(f'val_mse w/o sigmoid: {val_mse}')
                

            total_val_mse += val_mse.item()
            
            val_model = accelerator.gather(outputs.loss).sum()
            accelerator.print(f'loss from model val: {val_model}')
            total_model_loss += val_model.item()
            
            

        


        
        avg_mse_loss = total_val_mse / len(val_loader)  # 计算所有批次的平均MSE
        total_model_loss = total_model_loss/ len(val_loader)
        
        accelerator.print(f'Epoch {epoch} val {args.loss_func}: {total_model_loss}; MSE: {avg_mse_loss} ')
        if accelerator.is_local_main_process:
            writer.add_scalar(f"Validation {args.loss_func}", total_model_loss, epoch)  # 将平均MSE写入TensorBoard
            writer.add_scalar("Validation MSE", avg_mse_loss, epoch)  # 将平均MSE写入TensorBoard




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
        torch.save(score_state_dict, os.path.join(args.runs_dir,'last_score.pt'))


        # accelerator.save(net_dict,os.path.join(args.runs_dir,'last.pt'))
            # 保存具有最低验证损失的模型
        if total_model_loss < best_bce_loss:
            model_save_id = os.path.join(args.runs_dir,'best')
            
            
            best_bce_loss = total_model_loss
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
            model_save_id,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
                )

            # 假设 model 是你的模型实例，且 model.score 是需要保存的子模块
            score_state_dict = unwrapped_model.score.state_dict()
            print(score_state_dict)
            # 保存 state_dict 到文件
            torch.save(score_state_dict, os.path.join(args.runs_dir,'best_score.pt'))

        accelerator.print(f'第{epoch}轮 训练集loss:{avg_train_loss}，验证集loss:{total_model_loss}')
    
if accelerator.is_local_main_process:
    writer.close()