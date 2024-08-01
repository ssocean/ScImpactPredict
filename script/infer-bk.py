import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, XLNetModel
import pandas as pd
from torch.utils.data import DataLoader
import os

# 加载模型和分词器
checkpoint = "roberta-large"
tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
# model_path = 'roberta-large_0515.pth'
#
# # 确保模型文件存在
# if not os.path.exists(model_path):
#     print("Model file not found!")
#     exit()

model = RobertaForSequenceClassification.from_pretrained(checkpoint, num_labels=1)
# tokenizer = AutoTokenizer.from_pretrained("xlnet/xlnet-base-cased",num_labels=1)
# model = XLNetModel.from_pretrained("xlnet/xlnet-base-cased")
# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 加载数据
data_path = r'C:\Users\Ocean\Documents\GitHub\ScitePredict\tools\data_for_model_balanced.csv'
data = pd.read_csv(data_path)

# 定义 Dataset 类
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.titles = dataframe['title']
        self.abstracts = dataframe['abstract']
        self.labels = dataframe['TNCSI']
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = f"Title: {self.titles[idx]}\nAbstract: {self.abstracts[idx]}"
        inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# 创建数据集和数据加载器
dataset = TextDataset(data, tokenizer)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# # 推理和打印结果
# with torch.no_grad():
model.eval()
for batch in loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    # print(outputs)
    logits = outputs.logits.squeeze(0)
    prediction = logits.item()  # 将 logits 转换为单个数值
    # print(batch['input_ids'])
    gt = batch['labels'].item()  # 真实标签
    print(f'Predicted: {prediction}, GT: {gt}')
