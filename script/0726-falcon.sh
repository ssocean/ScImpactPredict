DATA_PATH="/home/u1120220285/ScitePredict/NAID/NAID_train_extrainfo.csv"
TEST_DATA_PATH="/home/u1120220285/ScitePredict/NAID/NAID_test_extrainfo.csv"
prompt=0
# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --batch_size 12 \
#     --learning_rate 1e-4 \
#     --data_path $DATA_PATH \
#     --test_data_path $TEST_DATA_PATH \
#     --runs_dir /home/u1120220285/ScitePredict/official_runs/mc_falcon \
#     --checkpoint  /home/u1120220285/falcon \
#     --target_modules "query_key_value,dense,dense_h_to_4h,dense_4h_to_h"

# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --learning_rate 1e-4 \
#     --data_path $DATA_PATH \
#     --test_data_path $TEST_DATA_PATH \
#     --runs_dir /home/u1120220285/ScitePredict/official_runs/mc_Phi3 \
#     --checkpoint /home/u1120220285/Phi-3-mini-4k-instruct \
#     --target_modules "qkv_proj"



OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --prompt_style 3 \
    --learning_rate 1e-4 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_2020_train_extrainfo.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_2020_test_extrainfo.csv\
    --runs_dir /home/u1120220285/ScitePredict/official_runs/TNCSI_2020

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --prompt_style 3 \
    --learning_rate 1e-4 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_2021_train_extrainfo.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_2021_test_extrainfo.csv\
    --runs_dir /home/u1120220285/ScitePredict/official_runs/TNCSI_2021

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --prompt_style 3 \
    --learning_rate 1e-4 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_2022_train_extrainfo.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_2022_test_extrainfo.csv\
    --runs_dir /home/u1120220285/ScitePredict/official_runs/TNCSI_2022


OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_2020_train_extrainfo.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_2020_test_extrainfo.csv\
    --runs_dir /home/u1120220285/ScitePredict/official_runs/TNCSI_2020_SP

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_2021_train_extrainfo.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_2021_test_extrainfo.csv\
    --runs_dir /home/u1120220285/ScitePredict/official_runs/TNCSI_2021_SP

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_2022_train_extrainfo.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_2022_test_extrainfo.csv\
    --runs_dir /home/u1120220285/ScitePredict/official_runs/TNCSI_2022_SP

python scancel.py