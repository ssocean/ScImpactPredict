# ------------------- Finetune from the designated weight path---------------------


# OMP_NUM_THREADS=1 accelerate launch train.py \
#     --total_epochs 20 --loss_func mse --data_style 4\
# # auto scancel

# -------------------   Misc  ----------------------

# ---------------------------------------------------
# Default Finetune CMD

OMP_NUM_THREADS=1 accelerate launch train.py \
    --total_epochs 30 \
    --loss_func mse \
    --data_style 5

# OMP_NUM_THREADS=1 accelerate launch train.py \
#     --total_epochs 10 \
#     --loss_func mse \
#     --runs_dir xxx
#     --data_style 1 

python scancel.py


