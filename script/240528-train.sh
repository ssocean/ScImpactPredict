# ------------------- Finetune from the designated weight path---------------------


# -------------------   Misc  ----------------------

# ---------------------------------------------------
# Default Finetune CMD


# OMP_NUM_THREADS=1 accelerate launch train.py \
#     --total_epochs 5 \

# OMP_NUM_THREADS=1 accelerate launch train.py \
#     --total_epochs 10 \

OMP_NUM_THREADS=1 accelerate launch train.py \
    --total_epochs 20 \

# auto scancel

python scancel.py
