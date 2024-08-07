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
import transformers.models.qwen2
from peft import AutoPeftModelForCausalLM,AutoPeftModelForSequenceClassification,AutoPeftModelForTokenClassification
from transformers import AutoTokenizer
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType



title = '''xxxx'''
abstract = ''' xxx '''




model_pth = r"xxx"
model = AutoPeftModelForSequenceClassification.from_pretrained(model_pth,num_labels=1, load_in_8bit=True,)
tokenizer = AutoTokenizer.from_pretrained(model_pth)

model = model.to("cuda")
model.eval()
# Default Prompt Template
text = f'''Given a certain paper, Title: {title}\n Abstract: {abstract}. \n Predict its normalized academic impact (between 0 and 1):'''
inputs = tokenizer(text, return_tensors="pt")

outputs = model(input_ids=inputs["input_ids"].to("cuda"))
print(outputs['logits'])
