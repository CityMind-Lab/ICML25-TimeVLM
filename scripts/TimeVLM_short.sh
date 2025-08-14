export TOKENIZERS_PARALLELISM=false
model_name=TimeVLM
vlm_type=clip
gpu=0
image_size=56
norm_const=0.4
three_channel_image=True
finetune_vlm=False
batch_size=32
num_workers=32
learning_rate=0.001
seq_len=512
train_epochs=15

# Create logs directory if it doesn't exist
if [ ! -d "logs" ]; then
    mkdir logs
fi

# Function to run short-term forecast experiment (M4)
run_short_term_experiment() {
    local seasonal_pattern=$1
    local periodicity=$2
    local d_model=$3
    local use_mem_gate=$4

    log_file="logs/${model_name}_m4_${seasonal_pattern}_${periodicity}.log"
    echo "Running short-term forecast: M4 ${seasonal_pattern}, periodicity=${periodicity}"

    python -u run.py \
      --task_name short_term_forecast \
      --is_training 1 \
      --root_path ./dataset/m4 \
      --seasonal_patterns $seasonal_pattern \
      --model_id m4_${seasonal_pattern} \
      --model $model_name \
      --data m4 \
      --features M \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --d_model $d_model \
      --des 'Exp' \
      --itr 1 \
      --loss 'SMAPE' \
      --image_size $image_size \
      --norm_const $norm_const \
      --periodicity $periodicity \
      --three_channel_image $three_channel_image \
      --finetune_vlm $finetune_vlm \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --num_workers $num_workers \
      --vlm_type $vlm_type \
      --dropout 0.1 \
      --use_mem_gate $use_mem_gate > $log_file
}

run_short_term_experiment 'Yearly' 1 128 True
run_short_term_experiment 'Quarterly' 4 128 True
run_short_term_experiment 'Monthly' 3 128 True
run_short_term_experiment 'Weekly' 4 128 True
run_short_term_experiment 'Daily' 1 128 True
run_short_term_experiment 'Hourly' 24 128 True