MODEL='checkpoints/redditcc_warmup_40'
DATADIR='dataset/wizard_kat'
OUT_PREFIX='wizardlr_kat'
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=1

for rat in 2 4 8 16
do
    OUTPUT=checkpoints/${OUT_PREFIX}_${rat}
    PREFIX=train_$rat
    python run_kat.py \
        --model_name_or_path $MODEL \
        --data_dir $DATADIR \
        --cache_dir 'cached' \
        --task 'Wizard_LowResource' \
        --train_prefix $PREFIX \
        --eval_prefix 'test_seen' \
        --max_source_length 256 \
        --max_target_length 64 \
        --max_kno_length 64 \
        --max_num_kno 40 \
        --do_train \
        --do_eval \
        --num_train_epochs 10 \
        --save_steps 9000 \
        --per_gpu_train_batch_size 16 \
        --per_gpu_eval_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --output_dir $OUTPUT \
        --overwrite_output_dir --fp16 
done

for rat in 2 4 8 16
do
    OUTPUT=checkpoints/${OUT_PREFIX}_${rat}
    PREFIX=train_$rat
    python run_kat.py \
        --model_name_or_path $MODEL \
        --data_dir $DATADIR \
        --cache_dir 'cached' \
        --task 'Wizard_LowResource' \
        --train_prefix $PREFIX \
        --eval_prefix 'test_unseen' \
        --max_source_length 256 \
        --max_target_length 64 \
        --max_kno_length 64 \
        --max_num_kno 40 \
        --do_eval \
        --num_train_epochs 3 \
        --save_steps 9000 \
        --per_gpu_train_batch_size 16 \
        --per_gpu_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --output_dir $OUTPUT \
        --overwrite_output_dir --fp16 
done