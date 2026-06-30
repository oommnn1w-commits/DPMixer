=export CUDA_VISIBLE_DEVICES=0
model_name=DPMixer


python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model HCAP1 \
  --data m4 \
  --features M \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 1024 \
  --d_ff 32 \
  --Kmax 30 \
  --NWs 15 10 5 2 1 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --itr 1 \
  --des 'Exp' \
  --train_epochs 50 \
  --patience 20 \
  --loss 'SMAPE'


python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Yearly' \
  --model_id m4_Yearly \
  --model HCAP1 \
  --data m4 \
  --features M \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 1024 \
  --d_ff 32 \
  --Kmax 10 \
  --NWs 5 2 1 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --itr 1 \
  --des 'Exp' \
  --train_epochs 50 \
  --patience 20 \
  --loss 'SMAPE'

  python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Quarterly' \
  --model_id m4_Quarterly \
  --model HCAP1 \
  --data m4 \
  --features M \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 1024 \
  --d_ff 32 \
  --Kmax 14 \
  --NWs 7 6 5 4 3 2 1 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --itr 1 \
  --des 'Exp' \
  --train_epochs 50 \
  --patience 20 \
  --loss 'SMAPE'


python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Daily' \
  --model_id m4_Daily \
  --model HCAP1 \
  --data m4 \
  --features M \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 1024 \
  --d_ff 32 \
  --Kmax 10 \
  --NWs 5 2 1 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --itr 1 \
  --des 'Exp' \
  --train_epochs 50 \
  --patience 20 \
  --loss 'SMAPE'

  python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Weekly' \
  --model_id m4_Weekly \
  --model HCAP1 \
  --data m4 \
  --features M \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 1024 \
  --d_ff 32 \
  --Kmax 10 \
  --NWs  4 2 1 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --itr 1 \
  --des 'Exp' \
  --train_epochs 50 \
  --patience 20 \
  --loss 'SMAPE'

  python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Hourly' \
  --model_id m4_Hourly \
  --model HCAP1 \
  --data m4 \
  --features M \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 1024 \
  --d_ff 32 \
  --Kmax 24 \
  --NWs 12 10 8 6 4 2 1 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --itr 1 \
  --des 'Exp' \
  --train_epochs 50 \
  --patience 20 \
  --loss 'SMAPE'

