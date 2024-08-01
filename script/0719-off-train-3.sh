OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 3 \
    --learning_rate 1e-4 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_ALL_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_ALL_test_uniform.csv \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/0719-ubdata-smoothl1 \
    --loss_func smoothl1

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 3 \
    --learning_rate 1e-4 \
    --data_path /home/u1120220285/ScitePredict/NAID/NAID_ALL_train.csv \
    --test_data_path /home/u1120220285/ScitePredict/NAID/NAID_ALL_test_uniform.csv \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/0719-ubdata-focalmse \
    --loss_func focalmse

python scancel.py
