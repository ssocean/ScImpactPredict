# ------------------- Finetune from the designated weight path---------------------


# OMP_NUM_THREADS=1 accelerate launch train.py \
#     --total_epochs 20 --loss_func mse --data_style 4\
# # auto scancel

# -------------------   Misc  ----------------------

# ---------------------------------------------------
# Default Finetune CMD
OMP_NUM_THREADS=1 accelerate launch train.py \
    --total_epochs 10 \
    --loss_func bce \
    --data_style 0

OMP_NUM_THREADS=1 accelerate launch train.py \
    --total_epochs 10 \
    --loss_func bce \
    --data_style 1 

OMP_NUM_THREADS=1 accelerate launch train.py \
    --total_epochs 10 \
    --loss_func l1 \
    --data_style 1 

OMP_NUM_THREADS=1 accelerate launch train.py \
    --total_epochs 10 \
    --loss_func mse \
    --data_style 1 

OMP_NUM_THREADS=1 accelerate launch train.py \
    --total_epochs 10 \
    --loss_func bce \
    --data_style 2 

OMP_NUM_THREADS=1 accelerate launch train.py \
    --total_epochs 10 \
    --loss_func bce \
    --data_style 3 

OMP_NUM_THREADS=1 accelerate launch train.py \
    --total_epochs 10 \
    --loss_func bce \
    --data_style 4 

OMP_NUM_THREADS=1 accelerate launch train.py \
    --total_epochs 10 \
    --loss_func bce \
    --data_style 5 

python scancel.py


