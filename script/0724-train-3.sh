DATA_PATH="ScImpactPredict/NAID/NAID_train_extrainfo_remove_locnot0.csv"
TEST_DATA_PATH="ScImpactPredict/NAID/NAID_test_extrainfo_remove_locnot0.csv"
prompt=0

# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --learning_rate 1e-4 \
#     --data_path $DATA_PATH \
#     --test_data_path $TEST_DATA_PATH \
#     --runs_dir ScImpactPredict/official_runs/prompt-8 \
#     --prompt_style 8 

# OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
#     --total_epochs 5 \
#     --learning_rate 1e-4 \
#     --data_path $DATA_PATH \
#     --test_data_path $TEST_DATA_PATH \
#     --runs_dir ScImpactPredict/official_runs/prompt-7 \
#     --prompt_style 7 

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir ScImpactPredict/official_runs/prompt-10_removeloc0_newRQM \
    --prompt_style 10

DATA_PATH="ScImpactPredict/NAID/NAID_train_extrainfo.csv"
TEST_DATA_PATH="ScImpactPredict/NAID/NAID_test_extrainfo.csv"
OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir ScImpactPredict/official_runs/prompt-12 \
    --prompt_style 12
python scancel.py
