
class Config_lm(object):
    def __init__(self,**args):
        self.dropout_in=0.3
        self.dropout_out=0.3
        self.batch_size=32
        self.d_model=512
        self.d_word_vec=128
        self.dev_src_file='data/yelp_long/dev_20.neg'
        self.dev_trg_file='data/yelp_long/dev_20_0.attr'
        self.eval_every=2500
        self.log_every=100
        self.max_len=10000
        self.src_vocab='data/yelp_long/text.vocab'
        self.style=0
        self.tie_weight=False
        self.train_src_file='data/yelp_long/train_20.neg'
        self.train_trg_file='data/yelp_long/train_20_0.attr'
        self.trg_vocab='data/yelp_long/attr.vocab'