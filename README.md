# This is a modification from Deep Latent Sequence Model
from the [paper](https://arxiv.org/abs/2002.03912): 

```
A Probabilistic Formulation of Unsupervised Text Style Transfer
Junxian He*, Xinyi Wang*, Graham Neubig, Taylor Berg-Kirkpatrick
ICLR 2020
```

This project is used to explore different back-translation sentiment-transfer methods
## Requirements

* Python 3
* PyTorch >= 1.0

## Data
Datasets used in sentiment transfer already exists in the data/yelp

Another Datasets for different lengths are provided in data/yelp_long

## Pretrained LMs and Classifiers


This approach requires pretrained LMs as priors for each domain during trainining, and an oracle classifier is required at test time to compute the accuracy for sentiment transfer tasks. One choice is to download the pretrained LMs and classifiers to reproduce the results in the paper

Download pretrained lms (located in folder `./pretrained_lm`):
```
python scripts/prepare_lm.py --dataset yelp
```

Download pretrained classifiers (located in folder `./pretrained_classifer`):
```
python scripts/prepare_classifier.py --dataset yelp
```

## Usage
Training:
```
CUDA_VISIBLE_DEVICES=xx bash scripts/yelp/train_yelp.sh
```

Eval:
```
CUDA_VISIBLE_DEVICES=xx bash scripts/yelp/eval_all.sh [model dir]
```

The evaluation command will report several evaluation metrics (e.g. accuracy, self-bleu, reference bleu, and ppl for sentiment transfer task) and also transfer the test sentences to another domain, transferred test sentences are saved in `[model dir]`.




## Train your own LMs and Classifiers
Another choice is to train the own LMs and Classifiers,especially for testing in yelp_long dataset.

Train LMs:

```
train_my_lm.sh
```

To run the code on your own dataset, you need to create a new configuration file in `./config/` folder to specifiy network hyperparameters and datapath. If the new config file is `./config/config_abc.py`, then `[dataset]` needs to be set as `abc` accordingly. Pretrained LMs are saved in `./pretrained_lm/[dataset]` folder.



Train LSTM classifiers: 
```
train_my_classifier.sh
```


## Acknowledgement

We appreciate the efforts to the original writer for their well-style codes. The original codes are in the following link:https://github.com/cindyxinyiwang/deep-latent-sequence-model.git


