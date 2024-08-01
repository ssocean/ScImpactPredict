DATA_PATH="/home/u1120220285/ScitePredict/NAID/NAID_train_extrainfo.csv"
TEST_DATA_PATH="/home/u1120220285/ScitePredict/NAID/NAID_test_extrainfo.csv"
prompt=0

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 10 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/prompt-14_ep10 \
    --prompt_style 14

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 10 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/prompt-15_ep10 \
    --prompt_style 15

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 10 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/prompt-16_ep10 \
    --prompt_style 16

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 10 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir /home/u1120220285/ScitePredict/official_runs/prompt-17_ep10 \
    --prompt_style 17
python scancel.py