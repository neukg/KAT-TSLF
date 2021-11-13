MODEL='checkpoints/redditcc_warmup_20'
DATADIR='dataset/cmudog_kat'
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=1

for rat in 2 4 8 16
do
    OUTPUT=checkpoints/cmudoglr_kat_$rat
    PREFIX=train_$rat
    python run_kat.py \
        --model_name_or_path $MODEL \
        --data_dir $DATADIR \
        --cache_dir 'cached' \
        --task 'CMUDoG_LowResource' \
        --train_prefix $PREFIX \
        --max_source_length 256 \
        --max_target_length 64 \
        --max_kno_length 256 \
        --max_num_kno 4 \
        --do_train \
        --do_eval \
        --num_train_epochs 5 \
        --save_steps 9000 \
        --per_gpu_train_batch_size 16 \
        --per_gpu_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --output_dir $OUTPUT \
        --overwrite_output_dir --fp16 --num_beams 2
done