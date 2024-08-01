OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test.csv \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/0719-lr1e-4_wd1e-5


OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test.csv \
    --warmup_ratio 0.15 \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/0719-lr1e-4_warmup-015

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 10 \
    --learning_rate 1e-4 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test.csv \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/0719-lr1e-4_ep10
python scancel.py
