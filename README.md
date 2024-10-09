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

## Quick Try (for most researchers )
First, pull the repo and type following commands in the console:
```
cd ScImpactPredict
pip install -r requirements.txt
```

Then, download the model weights and modify the path to the model's weights in the `single_pred.py` file, then type `python single_pred.py` in the console.

## Model Weights

We recommend downloading the LLaMA-3 model weights, which offer the best performance. 

To begin with LLaMA3, you should request access and download the LLaMA-3 pretrain [weights](https://huggingface.co/meta-llama/Meta-Llama-3-8B) at huggingface official sites.
Then, download the provided LLaMA-3 LoRA weights (runs_dir) [here](https://drive.google.com/file/d/13-ugXsm35AuzOBUlL6jPacY_z8qVIb7x/view?usp=sharing).

We also offer the weights of other models for download.

| LLMs    | Size | MAE   | NDCG  | Mem   | Download Link    |
|---------|------|-------|-------|-------|------------------|
| Phi-3   | 3.8B | 0.226 | 0.742 | 6.2GB | [Download](https://drive.google.com/file/d/13-ugXsm35AuzOBUlL6jPacY_z8qVIb7x/view?usp=sharing)    |
| Falcon  | 7B   | 0.231 | 0.740 | 8.9GB | [Download](https://drive.google.com/file/d/13-ugXsm35AuzOBUlL6jPacY_z8qVIb7x/view?usp=sharing)    |
| Qwen-2  | 7B   | 0.223 | 0.774 | 12.6GB| [Download](https://drive.google.com/file/d/13-ugXsm35AuzOBUlL6jPacY_z8qVIb7x/view?usp=sharing)    |
| Mistral | 7B   | 0.220 | 0.850 | 15.4GB| [Download](https://drive.google.com/file/d/13-ugXsm35AuzOBUlL6jPacY_z8qVIb7x/view?usp=sharing)    |
| Llama-3 | 8B   | 0.216 | 0.901 | 9.4GB | [Download](https://drive.google.com/file/d/13-ugXsm35AuzOBUlL6jPacY_z8qVIb7x/view?usp=sharing)    |

## Fine-tuning (to reproduce or further improve the performance)

For fine-tuning, you may manually modify the 'xxxForSequenceClassification' in the `transformers` package. Or follow the [instruction](https://huggingface.co/docs/transformers/v4.27.1/en/custom_models#using-a-model-with-custom-code) to trust remote code.
```
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        ...
        self.post_init()
        # Add codes here!
        self.loss_func = 'mse'
        self.sigmoid = nn.Sigmoid()
        ...
    def forward(...):
        ...
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
            elif self.loss_func == 'l1':
                loss_fct = L1Loss()
            elif self.loss_func == 'smoothl1':
                loss_fct = nn.SmoothL1Loss()
        
```

Then, prepare `train.sh` bash file like below:
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
Finally, type `sh train.sh` in the console. Wating for the training ends~

## Testing
Similar to fine-tuning, prepare `test.sh` as below:
```
python inference.py \
 --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
 --weight_dir path_to_runs_dir
```
Then, type `sh test.sh`.




## Compare with Previous Methods 
With a few adjustments based on your specific needs, it should work fine. Since these models train very quickly (less than a few minutes on a single RTX 3080), we won’t be providing the trained weights.

##### Repo Structure Description
Folders like furnace, database, and tools are used for building the NAID and TKPD datasets. They have no direct connection to training or inference.

### We are pretty confident in our methodology and experiments, and you should be able to achieve any of the performance reported in our paper within an acceptable margin.
