# export CUDA_VISIBLE_DEVICES=0

SCRIPT_ROOT=./scripts
SCRIPT=./scripts/yelp

# printf "\ntranslate reference test\n"
# ${SCRIPT_ROOT}/translate.sh $1 data/yelp/test_li.txt data/yelp/test_li.attr transfer_ref.txt

printf "\nreference BLEU score\n\n"
path=$1
#$1 translation path
./multi-bleu.perl data/yelp/dev_li.txt < $path

# printf "\ntest reference classification\n\n"
# $SCRIPT/yelp_classify_test.sh $path data/yelp/test_li.attr

# printf "\ntranslate test_0\n\n"
# ${SCRIPT_ROOT}/translate.sh $1 data/yelp/test_0.txt data/yelp/test_0.attr transfer_test0.txt
# printf "\ntranslate test_1\n\n"
# ${SCRIPT_ROOT}/translate.sh $1 data/yelp/test_1.txt data/yelp/test_1.attr transfer_test1.txt


# path0 and path1 are transfer_text style0 and style1
# path0=$1transfer_text_more0.txt
# path1=$1transfer_text_more1.txt
# path0=$1ppl_0900.txt
# path1=$1ppl_1900.txt
# printf "\nLM test_1\n\n"
# $SCRIPT/eval_lm.sh 0 pretrained_lm/yelp_style0/model.pt $path1 data/yelp/test_1.attr
# printf "\nLM test_0\n\n"
# $SCRIPT/eval_lm.sh 1 pretrained_lm/yelp_style1/model.pt $path0 data/yelp/test_0.attr

# printf "\ntranslate entire test\n\n"
# ${SCRIPT_ROOT}/translate.sh $1 data/yelp/test.txt data/yelp/test.attr transfer_entire_test.txt

# path=$1ppl_0900.txt
# printf "\entire test self BLEU score\n\n"
# ./multi-bleu.perl data/yelp/test.txt < $path

# printf "\nentire test classification\n\n"
# $SCRIPT/yelp_classify_test.sh $path data/yelp/test.attr