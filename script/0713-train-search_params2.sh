# 验证max norm有效性
# OMP_NUM_THREADS=1 accelerate launch train.py \
#     --base_lr 1e-4 \
#     --total_epochs 10 \
#     --loss_func mse \
#     --data_style 8 \
#     --train_ratio 0.8 \
#     --max_norm -1 \
#     --data_path /home/u1120220285/ScitePredict/NAID/NAID_test.csv
# 数据集不同年份
# OMP_NUM_THREADS=1 accelerate launch train.py \
#     --base_lr 1e-4 \
#     --total_epochs 10 \
#     --loss_func mse \
#     --data_style 8 \
#     --data_path /home/u1120220285/ScitePredict/NAID/NAID_2021_train.csv
OMP_NUM_THREADS=1 accelerate launch train.py \
    --learning_rate 5e-05 \
    --batch_size 20\
    --total_epochs 10 \
    --loss_func mse \
    --data_style 1 \
    --train_ratio 0.8\
    --max_norm -1.0\
    --data_path /home/u1120220285/ScitePredict/data/NAID_8000_2022and2023.csv


OMP_NUM_THREADS=1 accelerate launch train_legacy.py \
    --learning_rate 5e-05 \
    --batch_size 20\
    --total_epochs 10 \
    --loss_func mse \
    --data_style 1 \
    --data_path /home/u1120220285/ScitePredict/data/NAID_8000_2022and2023.csv

OMP_NUM_THREADS=1 accelerate launch train.py \
    --base_lr 1e-4 \
    --total_epochs 10 \
    --loss_func mse \
    --data_style 8 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_2022_train.csv


OMP_NUM_THREADS=1 accelerate launch train.py \
    --base_lr 1e-4 \
    --total_epochs 10 \
    --loss_func mse \
    --data_style 8 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_2023_train.csv



python scancel.py


