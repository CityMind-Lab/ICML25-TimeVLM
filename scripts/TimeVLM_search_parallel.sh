#!/bin/sh

export TOKENIZERS_PARALLELISM=false
model_name=TimeVLM
vlm_type=clip
image_size=56
norm_const=0.4
three_channel_image=True
finetune_vlm=False
batch_size=32
num_workers=32
learning_rate=0.001
seq_len=512
percent=0.1
train_epochs=15

# GPU configuration - you have 4 GPUs (0,1,2,3)
available_gpus="0 1 2 3"
max_parallel_jobs=4

# Create logs directory if it doesn't exist
if [ ! -d "logs" ]; then
    mkdir logs
fi

# Create hyperparameter search logs directory
if [ ! -d "logs/hyperparameter_search" ]; then
    mkdir -p logs/hyperparameter_search
fi

# Function to get next available GPU
get_next_gpu() {
    # Simple round-robin GPU assignment
    echo $1 | cut -d' ' -f$(($(($2 % 4)) + 1))
}

# Function to run a single experiment
run_single_experiment() {
    dset=$1
    data=$2
    n_vars=$3
    pred_len=$4
    periodicity=$5
    d_model=$6
    dropout=$7
    use_mem_gate=$8
    gpu_id=$9
    
    echo "Starting experiment on GPU $gpu_id: ${dset}_${pred_len}_dm${d_model}_do${dropout}_mem${use_mem_gate}"
    
    # Determine task name based on percent
    task_name="few_shot_forecast"
    if [ "$percent" = "1" ]; then
        task_name="long_term_forecast"
    fi
    
    # Create unique log file name
    log_file="logs/hyperparameter_search/${model_name}_${dset}_${seq_len}_${pred_len}_dm${d_model}_do${dropout}_mem${use_mem_gate}_${percent}p.log"
    
    # Run experiment
    python -u run.py \
      --task_name $task_name \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ${dset}.csv \
      --model_id ${dset}_${seq_len}_${pred_len}_dm${d_model}_do${dropout}_mem${use_mem_gate} \
      --model $model_name \
      --data $data \
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
      --des 'Parallel Hyperparameter Search' \
      --itr 1 \
      --gpu $gpu_id \
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
      --percent $percent > $log_file 2>&1
    
    echo "Experiment completed on GPU $gpu_id: ${dset}_${pred_len}_dm${d_model}_do${dropout}_mem${use_mem_gate}"
}

# Comprehensive hyperparameter search function with parallel execution
run_comprehensive_search_parallel() {
    dset=$1
    data=$2
    n_vars=$3
    pred_len=$4
    periodicity=$5
    
    echo "Starting comprehensive hyperparameter search for ${dset} with pred_len=${pred_len}"
    echo "================================================================"
    
    # Extended grid search parameters - d_model from 32 to 512
    d_models="32 64 128 256 512"
    dropouts="0.1 0.2 0.3"
    use_mem_gates="True False"
    
    # Results summary file
    summary_file="logs/hyperparameter_search/${dset}_${pred_len}_comprehensive_summary.txt"
    echo "Comprehensive Hyperparameter Search Results for ${dset} (pred_len=${pred_len})" > $summary_file
    echo "Date: $(date)" >> $summary_file
    echo "Parallel execution: $max_parallel_jobs jobs" >> $summary_file
    echo "================================================================" >> $summary_file
    echo "d_model | dropout | use_mem_gate | MSE | MAE | Best | Params | GPU" >> $summary_file
    echo "--------|---------|---------------|-----|-----|------|-------|-----" >> $summary_file
    
    best_mse=999999
    best_config=""
    experiment_count=0
    total_experiments=30
    running_jobs=0
    job_count=0
    
    # Create a temporary file to store running job PIDs
    temp_pid_file="/tmp/timevlm_jobs_$$"
    echo "" > $temp_pid_file
    
    for d_model in $d_models; do
        for dropout in $dropouts; do
            for use_mem_gate in $use_mem_gates; do
                experiment_count=$(($experiment_count + 1))
                echo "Experiment ${experiment_count}/${total_experiments}: d_model=${d_model}, dropout=${dropout}, use_mem_gate=${use_mem_gate}"
                
                # Get next available GPU
                gpu_id=$(get_next_gpu "$available_gpus" $job_count)
                
                # Start experiment in background
                run_single_experiment $dset $data $n_vars $pred_len $periodicity $d_model $dropout $use_mem_gate $gpu_id &
                job_pid=$!
                
                # Store job info
                echo "$job_pid $dset $pred_len $d_model $dropout $use_mem_gate $gpu_id" >> $temp_pid_file
                running_jobs=$(($running_jobs + 1))
                job_count=$(($job_count + 1))
                
                echo "Started experiment on GPU $gpu_id (PID: $job_pid)"
                
                # Wait if we've reached max parallel jobs
                if [ $running_jobs -ge $max_parallel_jobs ]; then
                    echo "Waiting for jobs to complete... (currently running: $running_jobs)"
                    wait
                    running_jobs=0
                    echo "All jobs completed, continuing..."
                fi
            done
        done
    done
    
    # Wait for remaining jobs to complete
    if [ $running_jobs -gt 0 ]; then
        echo "Waiting for remaining $running_jobs jobs to complete..."
        wait
        echo "All jobs completed!"
    fi
    
    # Process results
    echo "Processing results..."
    for d_model in $d_models; do
        for dropout in $dropouts; do
            for use_mem_gate in $use_mem_gates; do
                # Find the log file
                log_file="logs/hyperparameter_search/${model_name}_${dset}_${seq_len}_${pred_len}_dm${d_model}_do${dropout}_mem${use_mem_gate}_${percent}p.log"
                
                # Extract MSE and MAE from log file
                mse=$(grep "mse:" $log_file | tail -1 | awk '{print $2}' | sed 's/,//')
                mae=$(grep "mae:" $log_file | tail -1 | awk '{print $4}' | sed 's/,//')
                
                # Extract model parameters count
                params=$(grep "Learnable model parameters:" $log_file | awk '{print $4}' | sed 's/,//')
                
                # Get GPU ID from temp file
                gpu_used=$(grep "$dset $pred_len $d_model $dropout $use_mem_gate" $temp_pid_file | awk '{print $7}')
                
                # Check if extraction was successful
                if [ -n "$mse" ] && [ "$mse" != "" ]; then
                    echo "Results: MSE=${mse}, MAE=${mae}, Params=${params}, GPU=${gpu_used}"
                    
                    # Check if this is the best result so far
                    is_best=""
                    if [ $(echo "$mse < $best_mse" | bc -l) -eq 1 ]; then
                        best_mse=$mse
                        best_config="d_model=${d_model}, dropout=${dropout}, use_mem_gate=${use_mem_gate}"
                        is_best="*"
                        echo "New best result found!"
                    fi
                    
                    # Add to summary
                    echo "${d_model} | ${dropout} | ${use_mem_gate} | ${mse} | ${mae} | ${is_best} | ${params} | ${gpu_used}" >> $summary_file
                else
                    echo "Failed to extract results from log file"
                    echo "${d_model} | ${dropout} | ${use_mem_gate} | FAILED | FAILED |  | FAILED | ${gpu_used}" >> $summary_file
                fi
            done
        done
    done
    
    # Clean up temp file
    rm -f $temp_pid_file
    
    # Final summary
    echo "" >> $summary_file
    echo "Best Configuration: ${best_config}" >> $summary_file
    echo "Best MSE: ${best_mse}" >> $summary_file
    echo "================================================================" >> $summary_file
    
    echo "Comprehensive hyperparameter search completed for ${dset} (pred_len=${pred_len})"
    echo "Best configuration: ${best_config}"
    echo "Best MSE: ${best_mse}"
    echo "Results saved to: ${summary_file}"
    echo "================================================================"
    echo ""
}

# Dataset configurations - using simple space-separated strings
datasets="ETTh1 ETTh2 ETTm1 ETTm2 Electricity Traffic Weather"
data_types="ETTh1 ETTh2 ETTm1 ETTm2 custom custom custom"
n_vars_list="7 7 7 7 321 862 21"
periodicities="24 24 96 96 24 24 144"

# Prediction lengths to test
pred_lengths="96 192 336 720"

# Main execution
echo "Starting PARALLEL comprehensive hyperparameter search for ALL datasets and prediction lengths"
echo "This will test d_model from 32 to 512 for all combinations"
echo "Total experiments per dataset-pred_len: 5 × 3 × 2 = 30 combinations"
echo "Total datasets: 7"
echo "Total prediction lengths: 4"
echo "Total experiments: 7 × 4 × 30 = 840 experiments"
echo "Parallel execution: $max_parallel_jobs jobs simultaneously"
echo "Available GPUs: $available_gpus"
echo ""

# Counter for total experiments
total_experiments=0

# Run search for each dataset and prediction length
dataset_count=1
for dset in $datasets; do
    # Get corresponding data_type, n_vars, and periodicity
    data_type=$(echo $data_types | cut -d' ' -f$dataset_count)
    n_vars=$(echo $n_vars_list | cut -d' ' -f$dataset_count)
    periodicity=$(echo $periodicities | cut -d' ' -f$dataset_count)
    
    echo "Processing dataset: ${dset} (n_vars=${n_vars}, periodicity=${periodicity})"
    echo "================================================================"
    
    for pred_len in $pred_lengths; do
        echo "Testing ${dset} with pred_len=${pred_len}..."
        run_comprehensive_search_parallel $dset $data_type $n_vars $pred_len $periodicity
        total_experiments=$(($total_experiments + 30))
        echo "Completed ${total_experiments}/840 total experiments"
        echo ""
    done
    
    echo "Completed all prediction lengths for ${dset}"
    echo "================================================================"
    echo ""
    
    dataset_count=$(($dataset_count + 1))
done

echo "ALL COMPREHENSIVE HYPERPARAMETER SEARCHES COMPLETED!"
echo "Total experiments run: ${total_experiments}"
echo "Check logs/hyperparameter_search/ for detailed results"
echo "Each dataset-pred_len combination has a comprehensive summary file"
