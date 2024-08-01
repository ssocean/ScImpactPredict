# ------------------- Finetune from the designated weight path---------------------


# OMP_NUM_THREADS=1 accelerate launch train.py \
#     --total_epochs 20 --loss_func mse --data_style 4\
# # auto scancel

# -------------------   Misc  ----------------------

# ---------------------------------------------------
# Default Finetune CMD


OMP_NUM_THREADS=1 accelerate launch train.py \
    --total_epochs 20 \
    --loss_func mse 

# OMP_NUM_THREADS=1 accelerate launch train.py \
#     --total_epochs 20 \
 

# OMP_NUM_THREADS=1 accelerate launch train.py \
#     --total_epochs 20 \
#     --data_style 2 

# OMP_NUM_THREADS=1 accelerate launch train.py \
#     --total_epochs 20 \
#     --data_style 2 

# OMP_NUM_THREADS=1 accelerate launch train.py \
#     --total_epochs 20 \
#     --data_style 3

# OMP_NUM_THREADS=1 accelerate launch train.py \
#     --total_epochs 20 \
#     --data_style 4 


# python scancel.py
