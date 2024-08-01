DATA_PATH="/home/u1120220285/ScitePredict/NAID/NAID_train_extrainfo.csv"
TEST_DATA_PATH="/home/u1120220285/ScitePredict/NAID/NAID_test_extrainfo.csv"
prompt=0

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 10 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/mc_qwen7b_ep10 \
    --checkpoint /home/u1120220285/qwen-7B/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 10 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/mc_qwen-1-5b_ep10 \
    --checkpoint  /home/u1120220285/qwen-1.5B/models--Qwen--Qwen2-1.5B/snapshots/8a16abf2848eda07cc5253dec660bf1ce007ad7a

python scancel.py
