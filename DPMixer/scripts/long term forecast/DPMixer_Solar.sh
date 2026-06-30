#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

MODEL_NAME=DPMixer
ITR=1

# Solar dataset configuration
DATASET=Solar
FILE_NAME=solar_AL.csv
SEQ_LEN=96
ENC_IN=137
DEC_IN=137
C_OUT=137

# Model hyperparameters
D_MODEL=1024
E_LAYERS=2
BATCH_SIZE=512
LEARNING_RATE=0.001
KMAX=80
NWS_GROUPS=(40 35 30 25 20 15 10 5 2)

# ==============================================
# Forecasting horizon 96
# ==============================================
PRED_LEN=96
MODEL_ID=${DATASET}${SEQ_LEN}_${PRED_LEN}_dm${D_MODEL}_el${E_LAYERS}_bs${BATCH_SIZE}_lr${LEARNING_RATE}

echo "=============================================="
echo "Running Solar | pred_len=$PRED_LEN"
echo "=============================================="

python -u run.py \
    --is_training 1 \
    --label_len 0 \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --task_name long_term_forecast \
    --root_path ./dataset/ \
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
    --use_channel_mixing 1 \
    --itr $ITR

# ==============================================
# Forecasting horizon 192
# ==============================================
PRED_LEN=192
MODEL_ID=${DATASET}${SEQ_LEN}_${PRED_LEN}_dm${D_MODEL}_el${E_LAYERS}_bs${BATCH_SIZE}_lr${LEARNING_RATE}

echo "=============================================="
echo "Running Solar | pred_len=$PRED_LEN"
echo "=============================================="

python -u run.py \
    --is_training 1 \
    --label_len 0 \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --task_name long_term_forecast \
    --root_path ./dataset/ \
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
    --use_channel_mixing 1 \
    --itr $ITR

# ==============================================
# Forecasting horizon 336
# ==============================================
PRED_LEN=336
MODEL_ID=${DATASET}${SEQ_LEN}_${PRED_LEN}_dm${D_MODEL}_el${E_LAYERS}_bs${BATCH_SIZE}_lr${LEARNING_RATE}

echo "=============================================="
echo "Running Solar | pred_len=$PRED_LEN"
echo "=============================================="

python -u run.py \
    --is_training 1 \
    --label_len 0 \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --task_name long_term_forecast \
    --root_path ./dataset/ \
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
    --use_channel_mixing 1 \
    --itr $ITR

# ==============================================
# Forecasting horizon 720
# ==============================================
PRED_LEN=720
MODEL_ID=${DATASET}${SEQ_LEN}_${PRED_LEN}_dm${D_MODEL}_el${E_LAYERS}_bs${BATCH_SIZE}_lr${LEARNING_RATE}

echo "=============================================="
echo "Running Solar | pred_len=$PRED_LEN"
echo "=============================================="

python -u run.py \
    --is_training 1 \
    --label_len 0 \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --task_name long_term_forecast \
    --root_path ./dataset/ \
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
    --use_channel_mixing 1 \
    --itr $ITR

echo "Solar all experiments completed!"