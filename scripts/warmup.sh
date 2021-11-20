MODEL='./redditcc_base'
KNOMODEL='./bart_wikilm'
DATADIR='dataset'
export TOKENIZERS_PARALLELISM=true

OUTPUT='checkpoints/redditcc_stage2_40'
CUDA_VISIBLE_DEVICES=0 python run_kat.py \
    --model_name_or_path $MODEL \
    --kno_mlm_model_path $KNOMODEL \
    --data_dir $DATADIR \
    --cache_dir 'cached' \
    --train_prefix 'train_p1_n39.jsonl' \
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

OUTPUT='checkpoints/redditcc_stage2_20'
CUDA_VISIBLE_DEVICES=0 python run_kat.py \
    --model_name_or_path $MODEL \
    --kno_mlm_model_path $KNOMODEL \
    --data_dir $DATADIR \
    --cache_dir 'cached' \
    --train_prefix 'train_p1_n39.jsonl' \
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

OUTPUT='checkpoints/redditcc_stage2_4'
export TOKENIZERS_PARALLELISM=true
CUDA_VISIBLE_DEVICES=0 python run_kat.py \
    --model_name_or_path $MODEL \
    --kno_mlm_model_path $KNOMODEL \
    --data_dir $DATADIR \
    --cache_dir 'cached' \
    --train_prefix 'train_p1_n39.jsonl' \
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