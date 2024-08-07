DATA_PATH="ScImpactPredict/NAID/NAID_train_extrainfo.csv"
TEST_DATA_PATH="ScImpactPredict/NAID/NAID_test_extrainfo.csv"
prompt=0

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 6 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir ScImpactPredict/official_runs/prompt-91 \
    --prompt_style 91

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 6 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir ScImpactPredict/official_runs/prompt-13 \
    --prompt_style 13

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 6 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir ScImpactPredict/official_runs/prompt-14 \
    --prompt_style 14

python scancel.py
