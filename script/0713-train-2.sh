OMP_NUM_THREADS=1 accelerate launch train.py \
    --base_lr 1e-4 \
    --total_epochs 15 \
    --loss_func mse \
    --data_style 8 \
    --max_norm -1.0 \
    --seed -1 \
    --train_ratio 0.8 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_full.csv \

OMP_NUM_THREADS=1 accelerate launch train.py \
    --base_lr 1e-4 \
    --total_epochs 15 \
    --loss_func mse \
    --data_style 8 \
    --max_norm -1.0 \
    --seed -1 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_test.csv


OMP_NUM_THREADS=1 accelerate launch train.py \
    --base_lr 1e-4 \
    --total_epochs 15 \
    --loss_func mse \
    --data_style 8 \
    --max_norm -1.0 \
    --seed -1 \
    --weight_decay -1 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_2122_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_2122_test.csv

python scancel.py