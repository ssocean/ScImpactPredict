DATA_PATH="ScImpactPredict/NAID/NAID_train_extrainfo.csv"
TEST_DATA_PATH="ScImpactPredict/NAID/NAID_test_extrainfo.csv"
prompt=0


OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --batch_size 12 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir ScImpactPredict/official_runs/mc_mistral \
    --checkpoint  Mistral-7B-v0.1

OMP_NUM_THREADS=1 accelerate launch offcial_train.py \
    --total_epochs 5 \
    --prompt_style -1 \
    --learning_rate 1e-4 \
    --data_path $DATA_PATH \
    --test_data_path $TEST_DATA_PATH \
    --runs_dir ScImpactPredict/official_runs/TNCSI_all 
python scancel.py