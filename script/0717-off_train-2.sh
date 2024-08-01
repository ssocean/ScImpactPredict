# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --learning_rate 5e-5 \
#     --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train.csv \
#     --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test.csv \
#     --prompt_style 1 

# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --learning_rate 5e-5 \
#     --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train.csv \
#     --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test.csv \
#     --prompt_style 2 

# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --learning_rate 5e-5 \
#     --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train.csv \
#     --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test.csv \
#     --prompt_style 3
# python scancel.py

# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --learning_rate 5e-5 \
#     --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train.csv \
#     --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test.csv \
#     --weight_decay 1e-4

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 5e-5 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test.csv \
    --lora_r 8 \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/0717-l_r8

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 5e-5 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test.csv \
    --lora_r 32 \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/0717-l_r32

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 5e-5 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test.csv \
    --lora_alpha 16 \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/0717-l_a16

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 5e-5 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test.csv \
    --lora_dropout 0.1 \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/0717-l_d01
python scancel.py