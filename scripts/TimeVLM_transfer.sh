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

# Function to run zero-shot forecast experiment
run_zero_shot_experiment() {
    local source_data=$1
    local target_data=$2
    local n_vars=$3
    local pred_len=$4
    local d_model=$5
    local use_mem_gate=$6
    local periodicity=$7
    
    log_file="logs/${model_name}_${source_data}_${target_data}_${seq_len}_${pred_len}.log"
    echo "Running zero-shot experiment: source=${source_data}, target=${target_data}, seq_len=${seq_len}, pred_len=${pred_len}"

    python -u run.py \
      --task_name "zero_shot_forecast" \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ${source_data}.csv \
      --model_id ${source_data}_${target_data}_${seq_len}_${pred_len} \
      --model $model_name \
      --data $source_data \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in $n_vars \
      --dec_in $n_vars \
      --c_out $n_vars \
      --des 'Exp' \
      --itr 1 \
      --gpu $gpu \
      --use_amp \
      --train_epochs $train_epochs \
      --d_model $d_model \
      --image_size $image_size \
      --norm_const $norm_const \
      --periodicity $periodicity \
      --three_channel_image $three_channel_image \
      --finetune_vlm $finetune_vlm \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --num_workers $num_workers \
      --vlm_type $vlm_type \
      --use_mem_gate $use_mem_gate \
      --target_data $target_data \
      --target_root_path ./dataset/ \
      --target_data_path ${target_data}.csv > $log_file
}

# ETTh1 -> ETTh2
run_zero_shot_experiment "ETTh1" "ETTh2" 7 96 64 True 24
run_zero_shot_experiment "ETTh1" "ETTh2" 7 192 64 True 24
run_zero_shot_experiment "ETTh1" "ETTh2" 7 336 64 True 24
run_zero_shot_experiment "ETTh1" "ETTh2" 7 720 64 True 24

# ETTh1 -> ETTm2
run_zero_shot_experiment "ETTh1" "ETTm2" 7 96 64 True 24
run_zero_shot_experiment "ETTh1" "ETTm2" 7 192 64 True 24
run_zero_shot_experiment "ETTh1" "ETTm2" 7 336 64 True 24
run_zero_shot_experiment "ETTh1" "ETTm2" 7 720 64 True 24

# ETTh2 -> ETTh1
run_zero_shot_experiment "ETTh2" "ETTh1" 7 96 64 True 24
run_zero_shot_experiment "ETTh2" "ETTh1" 7 192 64 True 24
run_zero_shot_experiment "ETTh2" "ETTh1" 7 336 64 True 24
run_zero_shot_experiment "ETTh2" "ETTh1" 7 720 64 True 24

# ETTh2 -> ETTm2
run_zero_shot_experiment "ETTh2" "ETTm2" 7 96 64 True 24
run_zero_shot_experiment "ETTh2" "ETTm2" 7 192 64 True 24
run_zero_shot_experiment "ETTh2" "ETTm2" 7 336 64 True 24
run_zero_shot_experiment "ETTh2" "ETTm2" 7 720 64 True 24

# ETTm1 -> ETTh2
run_zero_shot_experiment "ETTm1" "ETTh2" 7 96 64 True 24
run_zero_shot_experiment "ETTm1" "ETTh2" 7 192 64 True 24
run_zero_shot_experiment "ETTm1" "ETTh2" 7 336 64 True 24
run_zero_shot_experiment "ETTm1" "ETTh2" 7 720 64 True 24

# ETTm1 -> ETTm2
run_zero_shot_experiment "ETTm1" "ETTm2" 7 96 64 True 24
run_zero_shot_experiment "ETTm1" "ETTm2" 7 192 64 True 24
run_zero_shot_experiment "ETTm1" "ETTm2" 7 336 64 True 24
run_zero_shot_experiment "ETTm1" "ETTm2" 7 720 64 True 24

# ETTm2 -> ETTh2
run_zero_shot_experiment "ETTm2" "ETTh2" 7 96 64 True 24
run_zero_shot_experiment "ETTm2" "ETTh2" 7 192 64 True 24
run_zero_shot_experiment "ETTm2" "ETTh2" 7 336 64 True 24
run_zero_shot_experiment "ETTm2" "ETTh2" 7 720 64 True 24

# ETTm2 -> ETTm1
run_zero_shot_experiment "ETTm2" "ETTm1" 7 96 64 True 24
run_zero_shot_experiment "ETTm2" "ETTm1" 7 192 64 True 24
run_zero_shot_experiment "ETTm2" "ETTm1" 7 336 64 True 24
run_zero_shot_experiment "ETTm2" "ETTm1" 7 720 64 True 24