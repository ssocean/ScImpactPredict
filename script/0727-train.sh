# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --prompt_style -3 \
#     --learning_rate 1e-4 \
#     --data_path /home/u1120220285/ScitePredict/NAID/NAID_2020_train_extrainfo.csv \
#     --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_2020_test_extrainfo.csv\
#     --runs_dir /home/u1120220285/ScitePredict/official_runs/TNCSI_2020_p3

# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --prompt_style -3 \
#     --learning_rate 1e-4 \
#     --data_path /home/u1120220285/ScitePredict/NAID/NAID_2021_train_extrainfo.csv \
#     --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_2021_test_extrainfo.csv\
#     --runs_dir /home/u1120220285/ScitePredict/official_runs/TNCSI_2021_p3

# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --prompt_style -3 \
#     --learning_rate 1e-4 \
#     --data_path /home/u1120220285/ScitePredict/NAID/NAID_2022_train_extrainfo.csv \
#     --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_2022_test_extrainfo.csv\
#     --runs_dir /home/u1120220285/ScitePredict/official_runs/TNCSI_2022_p3

# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --prompt_style -1 \
#     --learning_rate 1e-4 \
#     --data_path $DATA_PATH \
#     --test_data_path $TEST_DATA_PATH \
#     --runs_dir /home/u1120220285/ScitePredict/official_runs/TNCSI_all_p3

# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --prompt_style 3 \
#     --learning_rate 1e-4 \
#     --data_path /home/u1120220285/ScitePredict/NAID/NAID_2020_train_extrainfo.csv \
#     --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_2020_test_extrainfo.csv\
#     --runs_dir /home/u1120220285/ScitePredict/official_runs/TNCSI_2020_SP_p3

# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --prompt_style 3 \
#     --learning_rate 1e-4 \
#     --data_path /home/u1120220285/ScitePredict/NAID/NAID_2021_train_extrainfo.csv \
#     --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_2021_test_extrainfo.csv\
#     --runs_dir /home/u1120220285/ScitePredict/official_runs/TNCSI_2021_SP_p3

# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --prompt_style 3 \
#     --learning_rate 1e-4 \
#     --data_path /home/u1120220285/ScitePredict/NAID/NAID_2022_train_extrainfo.csv \
#     --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_2022_test_extrainfo.csv\
#     --runs_dir /home/u1120220285/ScitePredict/official_runs/TNCSI_2022_SP_p3
DATA_PATH="/home/u1120220285/ScitePredict/NAID/NAID_train_extrainfo.csv"
TEST_DATA_PATH="/home/u1120220285/ScitePredict/NAID/NAID_test_extrainfo.csv"
prompt=0

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 10 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/prompt-101_ep10 \
    --prompt_style 101

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 10 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/prompt-102_ep10 \
    --prompt_style 102

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 10 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/prompt-103_ep10 \
    --prompt_style 103

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 10 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/prompt-104_ep10 \
    --prompt_style 104

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 10 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/prompt-105_ep10 \
    --prompt_style 105

python scancel.py