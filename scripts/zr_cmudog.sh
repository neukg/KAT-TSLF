MODEL='checkpoints/redditcc_warmup_4'
DATADIR='dataset/cmudog_kat'
OUTPUT=$MODEL
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0

python run_kat.py \
    --model_name_or_path $MODEL \
    --data_dir $DATADIR \
    --cache_dir 'cached' \
    --task 'kat_zr_cmudog_beam1' \
    --max_source_length 256 \
    --max_target_length 64 \
    --max_kno_length 256 \
    --max_num_kno 20 \
    --do_eval \
    --num_train_epochs 3 \
    --save_steps 9000 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --output_dir $OUTPUT \
    --overwrite_output_dir --fp16 --num_beams 1