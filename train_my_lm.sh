#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
#SBATCH -t 0
#SBATCH --array=0-0%1
##SBATCH --nodelist=compute-0-7

python src/lm_lstm.py --dataset "yelp_long_30" \
--style 1 \
--dev_src_file "data/yelp_long/dev_30.pos" \
--dev_trg_file "data/yelp_long/dev_30_1.attr" \
--train_src_file "data/yelp_long/train_30.pos" \
--train_trg_file "data/yelp_long/train_30_1.attr" \
--src_vocab "data/yelp_long/text.vocab" \
--trg_vocab "data/yelp_long/attr.vocab" \
# --load_params \
# --params "/home/rtx2070s/exp/deep-latent-sequence-model/pretrained_lm/hparams_20.pt" 
# --src_vocab 'data/yelp_long/text_20.vocab' \
# --trg_vocab 'data/yelp_long/attr.vocab' \
# --train_src_file 'data/yelp_long/train_20.txt' \
# --train_trg_file 'data/yelp_long/train_20.attr' \
# --dev_src_file 'data/yelp_long/dev_20.txt' \
# --dev_trg_file 'data/yelp_long/train_20.attr' \