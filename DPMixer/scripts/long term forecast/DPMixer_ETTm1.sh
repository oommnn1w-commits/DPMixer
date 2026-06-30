#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

MODEL_NAME=DPMixer
ITR=1

# ETTm1 dataset configuration
DATASET=ETTm1
FILE_NAME=ETTm1.csv
SEQ_LEN=96
PRED_LEN=96
ENC_IN=7
DEC_IN=7
C_OUT=7

# Model hyperparameters
D_MODEL=1024
E_LAYERS=3
BATCH_SIZE=512
LEARNING_RATE=0.001
KMAX=80
NWS_GROUPS=(40 35 30 25 20 15 10)

# ==============================================
# Forecasting horizon 96
# ==============================================
PRED_LEN=96
MODEL_ID=${DATASET}${SEQ_LEN}_${PRED_LEN}_dm${D_MODEL}_el${E_LAYERS}_bs${BATCH_SIZE}_lr${LEARNING_RATE}

echo "Running ETTm1 | pred_len=96"
python -u run.py \
    --is_training 1 \
    --label_len 0 \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --task_name long_term_forecast \
    --root_path ./dataset/ETT-small/ \
    --data_path $FILE_NAME \
    --model_id $MODEL_ID \
    --model $MODEL_NAME \
    --data $DATASET \
    --features M \
    --seq_len $SEQ_LEN \
    --pred_len $PRED_LEN \
    --e_layers $E_LAYERS \
    --enc_in $ENC_IN \
    --dec_in $DEC_IN \
    --c_out $C_OUT \
    --d_model $D_MODEL \
    --d_ff $((D_MODEL*2)) \
    --NWs ${NWS_GROUPS[@]} \
    --Kmax $KMAX \
    --itr $ITR

# ==============================================
# Forecasting horizon 192
# ==============================================
PRED_LEN=192
MODEL_ID=${DATASET}${SEQ_LEN}_${PRED_LEN}_dm${D_MODEL}_el${E_LAYERS}_bs${BATCH_SIZE}_lr${LEARNING_RATE}

echo "Running ETTm1 | pred_len=192"
python -u run.py \
    --is_training 1 \
    --label_len 0 \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --task_name long_term_forecast \
    --root_path ./dataset/ETT-small/ \
    --data_path $FILE_NAME \
    --model_id $MODEL_ID \
    --model $MODEL_NAME \
    --data $DATASET \
    --features M \
    --seq_len $SEQ_LEN \
    --pred_len $PRED_LEN \
    --e_layers $E_LAYERS \
    --enc_in $ENC_IN \
    --dec_in $DEC_IN \
    --c_out $C_OUT \
    --d_model $D_MODEL \
    --d_ff $((D_MODEL*2)) \
    --NWs ${NWS_GROUPS[@]} \
    --Kmax $KMAX \
    --itr $ITR

# ==============================================
# Forecasting horizon 336
# ==============================================
NWS_GROUPS=(45 40 35 30 25 20 15 10)
PRED_LEN=336
MODEL_ID=${DATASET}${SEQ_LEN}_${PRED_LEN}_dm${D_MODEL}_el${E_LAYERS}_bs${BATCH_SIZE}_lr${LEARNING_RATE}

echo "Running ETTm1 | pred_len=336"
python -u run.py \
    --is_training 1 \
    --label_len 0 \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --task_name long_term_forecast \
    --root_path ./dataset/ETT-small/ \
    --data_path $FILE_NAME \
    --model_id $MODEL_ID \
    --model $MODEL_NAME \
    --data $DATASET \
    --features M \
    --seq_len $SEQ_LEN \
    --pred_len $PRED_LEN \
    --e_layers $E_LAYERS \
    --enc_in $ENC_IN \
    --dec_in $DEC_IN \
    --c_out $C_OUT \
    --d_model $D_MODEL \
    --d_ff $((D_MODEL*2)) \
    --NWs ${NWS_GROUPS[@]} \
    --Kmax $KMAX \
    --itr $ITR

# ==============================================
# Forecasting horizon 720
# ==============================================
NWS_GROUPS=(40 35 30 25 20)
PRED_LEN=720
MODEL_ID=${DATASET}${SEQ_LEN}_${PRED_LEN}_dm${D_MODEL}_el${E_LAYERS}_bs${BATCH_SIZE}_lr${LEARNING_RATE}

echo "Running ETTm1 | pred_len=720"
python -u run.py \
    --is_training 1 \
    --label_len 0 \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --task_name long_term_forecast \
    --root_path ./dataset/ETT-small/ \
    --data_path $FILE_NAME \
    --model_id $MODEL_ID \
    --model $MODEL_NAME \
    --data $DATASET \
    --features M \
    --seq_len $SEQ_LEN \
    --pred_len $PRED_LEN \
    --e_layers $E_LAYERS \
    --enc_in $ENC_IN \
    --dec_in $DEC_IN \
    --c_out $C_OUT \
    --d_model $D_MODEL \
    --d_ff $((D_MODEL*2)) \
    --NWs ${NWS_GROUPS[@]} \
    --Kmax $KMAX \
    --itr $ITR

echo "ETTm1 all experiments completed!"