MODEL='checkpoints/redditcc_stage2_4'
DATADIR='dataset/cmudog_kat'
OUTPUT='checkpoints/cmudog_kat'
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0

python run_kat.py \
    --model_name_or_path $MODEL \
    --data_dir $DATADIR \
    --cache_dir 'cached' \
    --task cmudog_full \
    --train_prefix 'train' \
    --eval_prefix 'test' \
    --max_source_length 256 \
    --max_target_length 64 \
    --max_kno_length 256 \
    --max_num_kno 4 \
    --do_train \
    --num_train_epochs 3 \
    --save_steps 9000 \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --output_dir $OUTPUT \
    --overwrite_output_dir --fp16 
for beams in 1 2 3 4
do
    echo Results beams $beams
    python run_kat.py \
        --model_name_or_path $MODEL \
        --data_dir $DATADIR \
        --cache_dir 'cached' \
        --task cmudog_full_b${beams} \
        --train_prefix 'train' \
        --eval_prefix 'test' \
        --max_source_length 256 \
        --max_target_length 64 \
        --max_kno_length 256 \
        --max_num_kno 4 \
        --do_eval \
        --num_train_epochs 3 \
        --save_steps 9000 \
        --per_gpu_train_batch_size 16 \
        --per_gpu_eval_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --output_dir $OUTPUT \
        --overwrite_output_dir --fp16 --num_beams ${beams}
done