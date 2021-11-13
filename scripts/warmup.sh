MODEL='pretrained/redditcc_base'
KNOMODEL='pretrained/bart_wikilm'
DATADIR='dataset/redditcc_full_warmup'

OUTPUT='checkpoints/redditcc_warmup_40'
export TOKENIZERS_PARALLELISM=true
CUDA_VISIBLE_DEVICES=0 python run_kat.py \
    --model_name_or_path $MODEL \
    --kno_mlm_model_path $KNOMODEL \
    --data_dir $DATADIR \
    --cache_dir 'cached' \
    --train_prefix 'train_40' \
    --task 'kat' \
    --init_kno_encoder \
    --max_source_length 256 \
    --max_target_length 64 \
    --max_kno_length 64 \
    --max_num_kno 40 \
    --do_train \
    --num_train_epochs 3 \
    --save_steps 4000 \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --output_dir $OUTPUT \
    --overwrite_output_dir --fp16

OUTPUT='checkpoints/redditcc_warmup_20'
export TOKENIZERS_PARALLELISM=true
CUDA_VISIBLE_DEVICES=0 python run_kat.py \
    --model_name_or_path $MODEL \
    --kno_mlm_model_path $KNOMODEL \
    --data_dir $DATADIR \
    --cache_dir 'cached' \
    --train_prefix 'train_20' \
    --task 'kat' \
    --init_kno_encoder \
    --max_source_length 256 \
    --max_target_length 64 \
    --max_kno_length 64 \
    --max_num_kno 20 \
    --do_train \
    --num_train_epochs 3 \
    --save_steps 4000 \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --output_dir $OUTPUT \
    --overwrite_output_dir --fp16

OUTPUT='checkpoints/redditcc_warmup_4'
export TOKENIZERS_PARALLELISM=true
CUDA_VISIBLE_DEVICES=0 python run_kat.py \
    --model_name_or_path $MODEL \
    --kno_mlm_model_path $KNOMODEL \
    --data_dir $DATADIR \
    --cache_dir 'cached' \
    --train_prefix 'train_4' \
    --task 'kat' \
    --init_kno_encoder \
    --max_source_length 256 \
    --max_target_length 64 \
    --max_kno_length 64 \
    --max_num_kno 4 \
    --do_train \
    --num_train_epochs 3 \
    --save_steps 4000 \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 64 \
    --gradient_accumulation_steps 4 \
    --output_dir $OUTPUT \
    --overwrite_output_dir --fp16