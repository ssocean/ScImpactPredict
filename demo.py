import torch.nn as nn
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer
import torch


# Warning, you may have to modify the "base_model_name_or_path" in adapter_config.json.
model_pth = r"path_to_the_weights_dir"
model = AutoPeftModelForSequenceClassification.from_pretrained(model_pth,num_labels=1, load_in_8bit=True,)
tokenizer = AutoTokenizer.from_pretrained(model_pth)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()


while True:
    title = input("Enter a title: ")
    abstract = input("Enter a abstract: ")
    title = title.replace("\n", "").strip()
    abstract = abstract.replace("\n", "").strip()

    # Default Prompt Template
    text = f'''Given a certain paper, Title: {title}\n Abstract: {abstract}. \n Predict its normalized academic impact (between 0 and 1):'''
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(input_ids=inputs["input_ids"].to("cuda"))


    # If you haven't modify the LLaMA code. (For most users)
    print(nn.Sigmoid()(outputs['logits']))

    # Else print(outputs['logits'])


