python inference.py --loss_func mse --data_style 88 \
 --data_path ScImpactPredict/NAID/NAID_012_test.csv \
 --test_ratio 0.9\
 --weight_dir ScImpactPredict/official_runs_0717-0719/0717-lr1e-4/checkpoint-440


 python inference.py --loss_func mse --data_style 88 \
 --data_path  ScImpactPredict/NAID/NAID_ALL_test_uniform.csv \
 --test_ratio 0.9\
 --weight_dir ScImpactPredict/official_runs_0717-0719/0717-lr1e-4/checkpoint-440