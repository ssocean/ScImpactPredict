

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train_extrainfo.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test_extrainfo.csv \
    --prompt_style 6 \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/0720-prompt6-smoothl1 \
    --loss_func smoothl1

# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --learning_rate 1e-4 \
#     --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train_extrainfo.csv \
#     --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test_extrainfo.csv \
#     --prompt_style 5 \
#     --runs_dir /home/u1120220285/ScitePredict/official_runs/0720-prompt5

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train_extrainfo.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test_extrainfo.csv \
    --prompt_style 6 \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/0720-prompt6

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 10 \
    --learning_rate 1e-4 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train_extrainfo.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test_extrainfo.csv \
    --prompt_style 4 \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/0720-prompt4_ep10


python scancel.py
