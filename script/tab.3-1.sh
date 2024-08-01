DATA_PATH="/home/u1120220285/ScitePredict/NAID/NAID_train_extrainfo.csv"
TEST_DATA_PATH="/home/u1120220285/ScitePredict/NAID/NAID_test_extrainfo.csv"
prompt=0

# py39

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/mc_LLAMA3 \

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/mc_mistral \
    --checkpoint  /home/u1120220285/Mistral-7B-v0.1

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/mc_qwen7b \
    --checkpoint /home/u1120220285/qwen-7B/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/mc_qwen-1-5b \
    --checkpoint  /home/u1120220285/qwen-1.5B/models--Qwen--Qwen2-1.5B/snapshots/8a16abf2848eda07cc5253dec660bf1ce007ad7a


OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/mc_qwen0-5b \
    --checkpoint /home/u1120220285/qwen-0.5B/models--Qwen--Qwen2-0.5B/snapshots/ff3a49fac17555b8dfc4db6709f480cc8f16a9fe

python scancel.py
