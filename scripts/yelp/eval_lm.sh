

python src/lm_lstm.py \
    --dataset yelp \
    --style $1 \
    --eval_from $2 \
    --test_src_file $3 \
    --test_trg_file $4 \
    --src_vocab "data/yelp/text.vocab" \
    --trg_vocab "data/yelp/attr.vocab" \
    --train_src_file "data/yelp/train.txt" \
    --train_trg_file "data/yelp/train.attr" 





    # batch_size=32, cuda=True, d_model=512, d_word_vec=128, dataset='yelp', decode=False, dev_src_file='outputs_yelp/yelp_wd0.0_wb0.0_ws0.0_an3_pool5_klw0.1_lr0.001_t0.01_lm_bt_hard_avglen/transfer_test1.txt', dev_trg_file='data/yelp/test_1.attr', device=device(type='cuda'), dropout_in=0.3, dropout_out=0.3, eval_every=2500, eval_from='pretrained_lm/yelp_style0/model.pt', log_every=100, max_len=10000, output='', output_dir='pretrained_lm/yelp_eval_style0/', seed=783435, shuffle_train=False, src_vocab='data/yelp/text.vocab', style=0, test_src_file='outputs_yelp/yelp_wd0.0_wb0.0_ws0.0_an3_pool5_klw0.1_lr0.001_t0.01_lm_bt_hard_avglen/transfer_test1.txt', test_trg_file='data/yelp/test_1.attr', tie_weight=False, train_src_file='data/yelp/train_0.txt', train_trg_file='data/yelp/train_0.attr', trg_vocab='data/yelp/attr.vocab')