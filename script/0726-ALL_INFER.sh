export CUDA_LAUNCH_BLOCKING=1
# TAB 3 & Fig.4
# python official_inference.py \
#  --total_epochs 1\
#  --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
#  --checkpoint ScImpactPredict/official_runs/mc_LLAMA3/checkpoint-395

# python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs/mc_qwen7b_ep10/checkpoint-632

# python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs/mc_falcon/checkpoint-525

# python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs/mc_LLAMA3/checkpoint-395

# python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs/mc_mistral/checkpoint-525

# python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs/mc_Phi3/checkpoint-395

# python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs/mc_qwen0-5b/checkpoint-395

# python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs/mc_qwen-1-5b_ep10/checkpoint-474

# Tab.6

# python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_2020_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs/TNCSI_2020/checkpoint-150

# python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_2020_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs/TNCSI_2020_SP/checkpoint-150



# python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_2021_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs/TNCSI_2021/checkpoint-140

# python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_2021_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs/TNCSI_2021_SP/checkpoint-140



#  python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_2022_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs/TNCSI_2022/checkpoint-110

# python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_2022_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs/TNCSI_2022_SP/checkpoint-110

#  python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs/TNCSI_all/checkpoint-395

# Tab.5 
#  python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs_0717-0719/0719-prompt1/checkpoint-440 \
#  --prompt_style 1

#  python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs_0717-0719/0719-prompt2/checkpoint-440\
#  --prompt_style 2

#  python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs_0717-0719/0719-prompt3/checkpoint-440\
#  --prompt_style 3

# APP Additional info
python inference.py --loss_func mse \
 --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
 --test_ratio 1.0 \
 --weight_dir ScImpactPredict/official_runs/prompt-14_ep10/checkpoint-790 \
 --prompt_style 14

python inference.py --loss_func mse \
 --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
 --test_ratio 1.0 \
 --weight_dir ScImpactPredict/official_runs/prompt-14_ep10/checkpoint-711 \
 --prompt_style 14
python inference.py --loss_func mse \
 --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
 --test_ratio 1.0 \
 --weight_dir ScImpactPredict/official_runs/prompt-14_ep10/checkpoint-632 \
 --prompt_style 14

python inference.py --loss_func mse \
 --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
 --test_ratio 1.0 \
 --weight_dir ScImpactPredict/official_runs/prompt-14_ep10/checkpoint-553 \
 --prompt_style 14

python inference.py --loss_func mse \
 --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
 --test_ratio 1.0 \
 --weight_dir ScImpactPredict/official_runs/prompt-14_ep10/checkpoint-474 \
 --prompt_style 14

python inference.py --loss_func mse \
 --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
 --test_ratio 1.0 \
 --weight_dir ScImpactPredict/official_runs/prompt-14_ep10/checkpoint-395 \
 --prompt_style 14
# python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs/prompt-15_ep10/checkpoint-790 \
#  --prompt_style 15

# python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs/prompt-16_ep10/checkpoint-790\
#  --prompt_style 16

# python inference.py --loss_func mse \
#  --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
#  --test_ratio 1.0 \
#  --weight_dir ScImpactPredict/official_runs/prompt-17_ep10/checkpoint-790 \
#  --prompt_style 17
