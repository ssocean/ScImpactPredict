# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --learning_rate 1e-4 \
#     --weight_decay 1e-4 \
#     --data_path ScImpactPredict/NAID/NAID_012_train_extrainfo.csv \
#     --test_data_path ScImpactPredict/NAID/NAID_012_train_extrainfo.csv \
#     --prompt_style 4 \
#     --runs_dir ScImpactPredict/official_runs/0719-prompt4


# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --learning_rate 1e-4 \
#     --weight_decay 1e-4 \
#     --data_path ScImpactPredict/NAID/NAID_012_train.csv \
#     --test_data_path ScImpactPredict/NAID/NAID_012_test.csv \
#     --prompt_style 3 \
#     --runs_dir ScImpactPredict/official_runs/0719-prompt3

# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --learning_rate 1e-4 \
#     --weight_decay 1e-4 \
#     --data_path ScImpactPredict/NAID/NAID_012_train.csv \
#     --test_data_path ScImpactPredict/NAID/NAID_012_test.csv \
#     --prompt_style 2 \
#     --runs_dir ScImpactPredict/official_runs/0719-prompt2

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --data_path ScImpactPredict/NAID/NAID_012_train.csv \
    --test_data_path ScImpactPredict/NAID/NAID_012_test.csv \
    --prompt_style 1 \
    --runs_dir ScImpactPredict/official_runs/0719-prompt1
OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --data_path ScImpactPredict/NAID/NAID_012_train.csv \
    --test_data_path ScImpactPredict/NAID/NAID_012_test.csv \
    --prompt_style 1 \
    --runs_dir ScImpactPredict/official_runs/0719-prompt5
python scancel.py
