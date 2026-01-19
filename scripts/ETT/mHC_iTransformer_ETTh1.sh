# 首先创建 logs 目录（如果不存在）
mkdir -p logs

# 定义模型名称
model_name=mHC_iTransformer

# --- 第一个实验 ---
model_id="${model_name}_ETTh1"
seq_len=96

for pred_len in 96 192
do
  log_file="logs/${model_id}_${seq_len}_${pred_len}.log"
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id $model_id \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --e_layers 4 \
    --enc_in 7 \
    --rate 4 \
    --lradj type3\
    --iter 20 \
    --alpha 0.1 \
    --beta 0.1\
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --d_model 256 \
    --d_ff 256 \
    --itr 3 > "$log_file" 2>&1
done