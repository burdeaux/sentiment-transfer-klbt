#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
#SBATCH -t 0
#SBATCH --array=0-0%1
##SBATCH --nodelist=compute-0-7

python src/main.py \
  --dataset yelp_long_30 \
  --clean_mem_every 5 \
  --reset_output_dir \
  --classifier_dir="pretrained_classifer/yelp_long_30" \
  --train_src_file data/yelp_long/train_20_l0.txt \
  --train_trg_file data/yelp_long/train_20_l0.attr \
  --dev_src_file data/yelp_long/dev_20_l0.txt \
  --dev_trg_file data/yelp_long/dev_20_l0.attr \
  --dev_trg_ref data/yelp_long/dev_20_l0.txt \
  --src_vocab  data/yelp_long/text.vocab \
  --trg_vocab  data/yelp_long/attr.vocab \
  --d_word_vec=512 \
  --d_model=512 \
  --log_every=100 \
  --eval_every=1000 \
  --ppl_thresh=10000 \
  --eval_bleu \
  --batch_size 32 \
  --valid_batch_size 32 \
  --patience 5 \
  --lr_dec 0.5 \
  --lr 0.001 \
  --dropout 0.3 \
  --max_len 10000 \
  --seed 0 \
  --beam_size 1 \
  --word_blank 0. \
  --word_dropout 0. \
  --word_shuffle 0. \
  --cuda \
  --anneal_epoch 3 \
  --max_pool_k_size 5 \
  --bt \
  --klw 0.1 \
  --lm \
  --avg_len \
  --model_type  seq2seq\
  --temperature 0.01 \
  --exp_num 108_20_2 \
  --bt_stop_grad 
  #lm_hard_ stop gr greedy when t=0.01,sample when t=1
  #gs_soft 是soft gumbel-softmax
  #btsg only bt_stop+bt+gumbelsoft



  

