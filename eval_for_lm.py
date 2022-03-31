import os
import sys
from src.utils import *
import warnings



def main():
    #ref_file is the dev txt,
    #valid_file is the valid generation directory(not txt)
    #eval_all need to put generation txt as test.txt and test.attr in the directory
    output='/home/rtx2070s/exp/deep-latent-sequence-model/'
    set_num=30
    ref_file='/home/rtx2070s/exp/deep-latent-sequence-model/data/yelp_long/dev_20_s.txt'
    valid_file='/home/rtx2070s/exp/deep-latent-sequence-model/outputs_transformer1108201.0yelp_long_30/yelp_long_30_wd0.0_wb0.0_ws0.0_an3_pool5_klw0.1_lr0.001_t0.01_lm_bt_hard_avglen'
    output_name='outputs_long'+'_'+str(set_num)
    dataset='yelp_long_30'
    
    output_dir=os.path.join(output, output_name)

    if not os.path.isdir(output_dir):
        print("-" * 80)
        print("Path {} does not exist. Creating.".format(output_dir))
        os.makedirs(output_dir)


    print("-" * 80)
    log_file = os.path.join(output_dir, "stdout")
    print("Logging to {}".format(log_file))

    
    sys.stdout = Logger(log_file)
    
    eval_all(ref_file,valid_file,dataset)
    



if __name__ == "__main__":
    
  main()