OMP_NUM_THREADS=1 accelerate launch train.py \
    --base_lr 1e-4 \
    --total_epochs 30 \
    --loss_func mse \
    --data_style 9 \
    --max_norm -1.0 \
    --seed -1 \
    --data_path /home/u1120220285/ScitePredict/runs/Jul16_16-31-32_gpu18/train_data.csv \
    --test_data_path /home/u1120220285/ScitePredict/runs/Jul16_16-31-32_gpu18/test_data.csv \
    --val_data_path /home/u1120220285/ScitePredict/runs/Jul16_16-31-32_gpu18/val_data.csv


python scancel.py
