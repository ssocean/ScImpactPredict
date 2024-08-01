# python official_inference.py \
#  --total_epochs 1\
#  --data_path /home/u1120220285/ScitePredict/NAID/NAID_test_extrainfo.csv \
#  --checkpoint /home/u1120220285/ScitePredict/official_runs/mc_LLAMA3/checkpoint-395

python inference.py --loss_func mse \
 --data_path //home/u1120220285/ScitePredict/NAID/NAID_test_extrainfo.csv \
 --test_ratio 1.0 \
 --weight_dir /home/u1120220285/ScitePredict/official_runs/mc_Phi3/checkpoint-395


 python inference.py --loss_func mse \
 --data_path //home/u1120220285/ScitePredict/NAID/NAID_test_extrainfo.csv \
 --test_ratio 1.0 \
 --weight_dir /home/u1120220285/ScitePredict/official_runs/mc_Phi3/checkpoint-395