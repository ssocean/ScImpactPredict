

OMP_NUM_THREADS=1 accelerate launch train.py \
    --base_lr 1e-3 \
    --total_epochs 10 \
    --loss_func mse \
    --data_style 8 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_test.csv

OMP_NUM_THREADS=1 accelerate launch train.py \
    --base_lr 1e-4 \
    --total_epochs 10 \
    --loss_func mse \
    --data_style 8 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_test.csv

OMP_NUM_THREADS=1 accelerate launch train.py \
    --base_lr 2e-4 \
    --total_epochs 10 \
    --loss_func mse \
    --data_style 8 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_test.csv

OMP_NUM_THREADS=1 accelerate launch train.py \
    --base_lr 5e-5 \
    --total_epochs 10 \
    --loss_func mse \
    --data_style 8 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_test.csv


python scancel.py


