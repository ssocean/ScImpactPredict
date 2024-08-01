# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --learning_rate 1e-2 \
#     --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train.csv \
#     --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test.csv

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-3 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test.csv \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/0717-lr1e-3

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test.csv \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/0717-lr1e-4

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 5e-5 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test.csv \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/0717-lr5e-5

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 5e-5 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_012_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_012_test.csv \
    --warmup_ratio 0.05 \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/0717-warmup-005

python scancel.py
