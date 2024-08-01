

OMP_NUM_THREADS=1 accelerate launch train.py \
    --base_lr 1e-4 \
    --total_epochs 30 \
    --loss_func mse \
    --data_style 6 \
    --max_norm -1.0 \
    --seed -1 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_train.csv

OMP_NUM_THREADS=1 accelerate launch train.py \
    --base_lr 1e-4 \
    --total_epochs 30 \
    --loss_func mse \
    --data_style 7 \
    --max_norm -1.0 \
    --seed -1 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_train.csv


python scancel.py


