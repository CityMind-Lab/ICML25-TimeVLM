export TOKENIZERS_PARALLELISM=false
model_name=TimeVLM
vlm_type=clip
gpu=1
image_size=56
periodicity=144
norm_const=0.4
three_channel_image=True
finetune_vlm=False
batch_size=32
num_workers=32
learning_rate=0.001
percent=0.1
seq_len=512

python -u run.py \
  --task_name few_shot_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_512_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --gpu $gpu \
  --use_amp \
  --image_size $image_size \
  --norm_const $norm_const \
  --periodicity $periodicity \
  --three_channel_image $three_channel_image \
  --finetune_vlm $finetune_vlm \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --num_workers $num_workers \
  --vlm_type $vlm_type \
  --percent $percent

python -u run.py \
  --task_name few_shot_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_512_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --gpu $gpu \
  --use_amp \
  --image_size $image_size \
  --norm_const $norm_const \
  --periodicity $periodicity \
  --three_channel_image $three_channel_image \
  --finetune_vlm $finetune_vlm \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --num_workers $num_workers \
  --vlm_type $vlm_type \
  --percent $percent

python -u run.py \
  --task_name few_shot_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_512_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --gpu $gpu \
  --use_amp \
  --image_size $image_size \
  --norm_const $norm_const \
  --periodicity $periodicity \
  --three_channel_image $three_channel_image \
  --finetune_vlm $finetune_vlm \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --num_workers $num_workers \
  --vlm_type $vlm_type \
  --percent $percent

python -u run.py \
  --task_name few_shot_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_512_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --gpu $gpu \
  --use_amp \
  --image_size $image_size \
  --norm_const $norm_const \
  --periodicity $periodicity \
  --three_channel_image $three_channel_image \
  --finetune_vlm $finetune_vlm \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --num_workers $num_workers \
  --vlm_type $vlm_type \
  --percent $percent