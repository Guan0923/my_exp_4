# 首先创建 logs 目录（如果不存在）
mkdir -p logs

# 定义模型名称
model_name=iTransformer

# --- 第一个实验 ---
model_id="ETTh1_96_96"
seq_len=96
log_file="logs/${model_id}_${seq_len}.log"

# 将 echo 的信息和命令执行结果都追加到日志文件
echo "Running: $model_id, Seq Len: $seq_len" >> $log_file
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id $model_id \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1 \
  --gpu 0 >> $log_file 2>&1

# --- 第二个实验 ---
model_id="ETTh1_96_192"
seq_len=96
log_file="logs/${model_id}_${seq_len}.log"

echo "Running: $model_id, Seq Len: $seq_len" >> $log_file
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id $model_id \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --itr 1 \
  --gpu 0 >> $log_file 2>&1

# --- 第三个实验 ---
model_id="ETTh1_96_336"
seq_len=96
log_file="logs/${model_id}_${seq_len}.log"

echo "Running: $model_id, Seq Len: $seq_len" >> $log_file
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id $model_id \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --gpu 0 >> $log_file 2>&1

# --- 第四个实验 ---
model_id="ETTh1_96_720"
seq_len=96
log_file="logs/${model_id}_${seq_len}.log"

echo "Running: $model_id, Seq Len: $seq_len" >> $log_file
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id $model_id \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --rate 4 \
  --iter 20 \
  --seq_len $seq_len \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --gpu 0 >> $log_file 2>&1