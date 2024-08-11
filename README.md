# From Words to Worth: Newborn Article Impact Prediction with LLM


<p align="center">
  <img src="img\model.png" alt="icon" width="65%">
</p>

<h1 align="center">
  LLM as Article Impact Predictor
</h1>

### [Early Access Version]
###### This [paper](https://arxiv.org/abs/2408.03934?context=cs.CL) is currently under peer review. The code might change frequently. We are currently experiencing a severe staff shortage. If you encounter any issues during the replication process, please feel free to contact us through an issue or via email：oceanytech@gmail.com.

<!-- If you have any issues, feel free to reach out via Email: oceanytech@gmail.com or open an issue in the repository. -->

## Introduction

This repository contains the official implementation for the paper **"From Words to Worth: Newborn Article Impact Prediction with LLM"**. The tool is designed to PEFT the LLMs for the prediction of the future impact.

## Installation
At this Early Access stage，installation could be a little bit complicated. Sorry for any inconvenience.

First, you need pull the repo and type following commands in the console:
```
cd ScImpactPredict
pip install -r requirements.txt
```
Second, you have to manully modify the 'xxxForSequenceClassification' in the `transformers` package.
```
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)
        self.post_init()
        # Add codes here!
        self.loss_func = 'mse'
        self.sigmoid = nn.Sigmoid()
        ...
    def forward(...):
        ...
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        # Add codes here!
        if not self.loss_func == 'bce':
            logits = self.sigmoid(logits)
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        ...
        # Add codes here!
        if self.config.problem_type == "regression":
            if self.loss_func == 'bce':
                loss_fct = BCEWithLogitsLoss()
            elif self.loss_func == 'mse':
                loss_fct = MSELoss()
            # loss_fct = MSELoss()
            elif self.loss_func == 'l1':
                loss_fct = L1Loss()
            elif self.loss_func == 'smoothl1':
                loss_fct = nn.SmoothL1Loss()
        
```
## Fine-tuning
Prepare `train.sh` bash file like below:
```
DATA_PATH="ScImpactPredict/NAID/NAID_train_extrainfo.csv"
TEST_DATA_PATH="ScImpactPredict/NAID/NAID_test_extrainfo.csv"

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir ScImpactPredict/official_runs/LLAMA3 \
    --checkpoint  path_to_huggingface_LLaMA3
```
Then, type `sh train.sh` in the console. Wating for the training ends~

## Testing (batch)
Similar to Fine-tuning, prepare `test.sh` as below:
```
python inference.py \
 --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
 --weight_dir path_to_runs_dir
```
Then, type `sh test.sh`.

## Testing (single article)
Just modified the `single_pred.py` file, then type `python single_pred.py`.

## Model Weights
First, apply and download the LLaMA-3 pretrain [weights](https://huggingface.co/meta-llama/Meta-Llama-3-8B) at huggingface official sites.
Then, download the provided LoRA weights (runs_dir) [here](https://drive.google.com/file/d/13-ugXsm35AuzOBUlL6jPacY_z8qVIb7x/view?usp=sharing).

## Compare with Previos Methods 
With a few adjustments based on your specific needs, it should work fine. Since these models train very quickly (less than few minutes on a single RTX 3080), we won’t be providing the trained models.

### We are pretty confident in our methodology and experiments, and you should be able to achieve any of the performance reported in our paper within acceptable margin.
