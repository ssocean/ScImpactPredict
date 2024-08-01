OMP_NUM_THREADS=1 accelerate launch train.py \
    --base_lr 1e-4 \
    --total_epochs 5 \
    --loss_func mse \
    --data_style 8 \
    --warmup_ratio 0.03 \
    --weight_decay 1e-4 \
    --max_norm -1.0 \
    --seed -1 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_test.csv

OMP_NUM_THREADS=1 accelerate launch train.py \
    --total_epochs 5 \
    --loss_func mse \
    --data_style 8 \
    --base_lr 1e-3 \
    --warmup_ratio 0.03 \
    --weight_decay 1e-4 \
    --max_norm -1.0 \
    --seed -1 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_test.csv

OMP_NUM_THREADS=1 accelerate launch train.py \
    --total_epochs 5 \
    --loss_func mse \
    --data_style 8 \
    --base_lr 5e-5 \
    --warmup_ratio 0.03 \
    --weight_decay 1e-4 \
    --max_norm -1.0 \
    --seed -1 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_test.csv


OMP_NUM_THREADS=1 accelerate launch train.py \
    --total_epochs 5 \
    --loss_func mse \
    --data_style 8 \
    --base_lr 1e-4 \
    --warmup_ratio 0.1 \
    --weight_decay 1e-4 \
    --max_norm -1.0 \
    --seed -1 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_test.csv


# OMP_NUM_THREADS=1 accelerate launch train.py \
#     --total_epochs 5 \
#     --loss_func mse \
#     --data_style 8 \
#     --base_lr 1e-3 \
#     --warmup_ratio 0.03 \
#     --weight_decay 1e-5 \
#     --max_norm -1.0 \
#     --seed -1 \
#     --data_path /home/u1120220285/ScitePredict/NAID/NAID_train.csv \
#     --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_test.csv


# OMP_NUM_THREADS=1 accelerate launch train.py \
#     --total_epochs 5 \
#     --loss_func mse \
#     --data_style 8 \
#     --base_lr 1e-3 \
#     --warmup_ratio 0.03 \
#     --weight_decay 1e-2 \
#     --max_norm -1.0 \
#     --seed -1 \
#     --data_path /home/u1120220285/ScitePredict/NAID/NAID_train.csv \
#     --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_test.csv
python scancel.py