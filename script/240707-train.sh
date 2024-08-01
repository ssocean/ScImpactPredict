# ------------------- Finetune from the designated weight path---------------------


# OMP_NUM_THREADS=1 accelerate launch train.py \
#     --total_epochs 20 --loss_func mse --data_style 4\
# # auto scancel

# -------------------   Misc  ----------------------
# 
# ---------------------------------------------------
# Default Finetune CMD

# OMP_NUM_THREADS=1 accelerate launch train.py \
#     --total_epochs 10 \
#     --loss_func mse \
#     --data_style 1 \
#     --data_path /home/u1120220285/ScitePredict/data/NAID_8000_2022.csv

OMP_NUM_THREADS=1 accelerate launch train.py \
    --total_epochs 10 \
    --loss_func mse \
    --data_style 1 \
    --data_path /home/u1120220285/ScitePredict/data/NAID_2022_train.csv

OMP_NUM_THREADS=1 accelerate launch train.py \
    --total_epochs 10 \
    --loss_func mse \
    --data_style 1 \
    --data_path /home/u1120220285/ScitePredict/data/NAID_2023_train.csv


python scancel.py


