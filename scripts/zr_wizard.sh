MODEL='checkpoints/redditcc_stage2_20'
DATADIR='dataset/wizard_kat'
OUTPUT=$MODEL
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0
python run_kat.py \
    --model_name_or_path $MODEL \
    --data_dir $DATADIR \
    --cache_dir 'cached' \
    --task 'kat_beam5' \
    --eval_prefix 'test_seen' \
    --max_source_length 256 \
    --max_target_length 64 \
    --max_kno_length 64 \
    --max_num_kno 40 \
    --do_eval \
    --num_train_epochs 3 \
    --save_steps 9000 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --output_dir $OUTPUT \
    --overwrite_output_dir --fp16 --num_beams 1

python run_kat.py \
    --model_name_or_path $MODEL \
    --data_dir $DATADIR \
    --cache_dir 'cached' \
    --task 'kat_beam5' \
    --eval_prefix 'test_unseen' \
    --max_source_length 256 \
    --max_target_length 64 \
    --max_kno_length 64 \
    --max_num_kno 40 \
    --do_eval \
    --num_train_epochs 3 \
    --save_steps 9000 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --output_dir $OUTPUT \
    --overwrite_output_dir --fp16 --num_beams 1