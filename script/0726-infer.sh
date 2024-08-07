# python official_inference.py \
#  --total_epochs 1\
#  --data_path ScImpactPredict/NAID/NAID_test_extrainfo.csv \
#  --checkpoint ScImpactPredict/official_runs/mc_LLAMA3/checkpoint-395

python inference.py --loss_func mse \
 --data_path /ScImpactPredict/NAID/NAID_test_extrainfo.csv \
 --test_ratio 1.0 \
 --weight_dir ScImpactPredict/official_runs/mc_Phi3/checkpoint-395


 python inference.py --loss_func mse \
 --data_path /ScImpactPredict/NAID/NAID_test_extrainfo.csv \
 --test_ratio 1.0 \
 --weight_dir ScImpactPredict/official_runs/mc_Phi3/checkpoint-395