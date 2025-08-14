export TOKENIZERS_PARALLELISM=false
model_name=TimeVLM
vlm_type=clip
gpu=1
image_size=56
norm_const=0.4
three_channel_image=True
finetune_vlm=False
batch_size=32
num_workers=32
learning_rate=0.001
seq_len=512
percent=1
train_epochs=15

# Create logs directory if it doesn't exist
if [ ! -d "logs" ]; then
    mkdir logs
fi

# Supports both few-shot (percent < 1.0) and full-shot (percent = 1.0)
run_experiment() {
    local dset=$1
    local data=$2
    local n_vars=$3
    local pred_len=$4
    local d_model=$5
    local use_mem_gate=$6
    local periodicity=$7
    local dropout=$8

    # Determine task name based on percent
    local task_name="few_shot_forecast"
    if [ "$percent" = "1" ]; then
        task_name="long_term_forecast"
    fi

    log_file="logs/${model_name}_${dset}_${seq_len}_${pred_len}_${percent}p.log"
    echo "Running experiment: dataset=${dset}, seq_len=${seq_len}, pred_len=${pred_len}, percent=${percent}"

    python -u run.py \
      --task_name $task_name \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ${dset}.csv \
      --model_id ${dset}_${seq_len}_${pred_len} \
      --model $model_name \
      --data ${data} \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --d_model $d_model \
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
      --dropout $dropout \
      --percent $percent > $log_file
}

# ETTh1, n_vars=7, periodicity=24
run_experiment ETTh1 ETTh1 7 96 64 False 24 0.2
run_experiment ETTh1 ETTh1 7 192 64 False 24 0.2
run_experiment ETTh1 ETTh1 7 336 64 True 24 0.2
run_experiment ETTh1 ETTh1 7 720 64 False 24 0.2

# ETTh2, n_vars=7, periodicity=24
run_experiment ETTh2 ETTh2 7 96 64 False 24 0.2
run_experiment ETTh2 ETTh2 7 192 64 False 24 0.2
run_experiment ETTh2 ETTh2 7 336 64 False 24 0.2
run_experiment ETTh2 ETTh2 7 720 64 False 24 0.2

# ETTm1, n_vars=7, periodicity=96
run_experiment ETTm1 ETTm1 7 96 64 True 96
run_experiment ETTm1 ETTm1 7 192 64 True 96
run_experiment ETTm1 ETTm1 7 336 32 True 96
run_experiment ETTm1 ETTm1 7 720 32 True 96

# ETTm2, n_vars=7, periodicity=96
run_experiment ETTm2 ETTm2 7 96 32 True 96 0.2
run_experiment ETTm2 ETTm2 7 192 32 True 96 0.2
run_experiment ETTm2 ETTm2 7 336 32 True 96 0.2
run_experiment ETTm2 ETTm2 7 720 32 True 96 0.2

# Electricity, n_vars=321, periodicity=24
run_experiment Electricity custom 321 96 128 True 24 0.3
run_experiment Electricity custom 321 192 128 True 24 0.3
run_experiment Electricity custom 321 336 256 True 24 0.3
run_experiment Electricity custom 321 720 512 True 24 0.3

# Traffic, n_vars=862, periodicity=24
run_experiment Traffic custom 862 96 128 True 24 0.1
run_experiment Traffic custom 862 192 128 True 24 0.1
run_experiment Traffic custom 862 336 256 True 24 0.1
run_experiment Traffic custom 862 720 512 True 24 0.1

# Weather, n_vars=21, periodicity=144
run_experiment Weather custom 21 96 64 True 144 0.1
run_experiment Weather custom 21 192 64 True 144 0.1
run_experiment Weather custom 21 336 128 True 144 0.1
run_experiment Weather custom 21 720 64 True 144 0.1