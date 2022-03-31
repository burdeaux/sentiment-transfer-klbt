import os
import sys
import time
import gc
import re
import subprocess

from datetime import datetime

import numpy as np
from prompt_toolkit import formatted_text
import torch
import torch.nn as nn
import torch.nn.init as init

def memReport():
  for obj in gc.get_objects():
    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
      print(type(obj), obj.size())

def reorder(x, index):
  """original x is reordered in terms of index to get x,
  this function is to recover original index

  Args:
    x: list
    index: numpy array, index[i] == j means the ith element
           in x was located at j originally
  """

  assert(len(x) == len(index))
  new_x = [0 for _ in range(len(x))]

  for i, j in enumerate(index):
    new_x[j] = x[i]

  return new_x

def get_criterion(hparams):
  loss_reduce = False
  crit = nn.CrossEntropyLoss(ignore_index=hparams.pad_id, size_average=False, reduce=loss_reduce)
  if hparams.cuda:
    crit = crit.cuda()
  return crit

def eval_for_bleu(dev_trg_ref,valid_hyp_file):
  ref_file=dev_trg_ref #original
  
  bleu_str = subprocess.getoutput(
      "./multi-bleu.perl {0} < {1}".format(ref_file, valid_hyp_file))
  log_string = "test_bleu_score"
  log_string += "\n{}".format(bleu_str)
  bleu_str = bleu_str.split('\n')[-1].strip()
  reg = re.compile("BLEU = ([^,]*).*")
  try:
    valid_bleu = float(reg.match(bleu_str).group(1))
  except:
    valid_bleu = 0.
  log_string += " val_bleu={0:<.2f}".format(valid_bleu)
  print(log_string)

  return valid_bleu

def eval_all(dev_trg_ref,valid_hyp_root,dataset):
  ref_file=dev_trg_ref #original or reference text
  val_file_root=valid_hyp_root #translate text
  valid_hyp_file=os.path.join(val_file_root,"test.txt")
  print('reference file is',ref_file,"\n")
  print('test_file is',valid_hyp_file,'\n')
 
  
  #bleu
  bleu_str = subprocess.getoutput(
      "./multi-bleu.perl {0} < {1}".format(ref_file, valid_hyp_file))
  log_string = "test_bleu_score"
  log_string += "\n{}".format(bleu_str)
  bleu_str = bleu_str.split('\n')[-1].strip()
  reg = re.compile("BLEU = ([^,]*).*")
  try:
    valid_bleu = float(reg.match(bleu_str).group(1))
  except:
    valid_bleu = 0.
  log_string += " val_bleu={0:<.2f}".format(valid_bleu)
  print(log_string)

  #ppl
  #seperate neg and pos for ppl
  f_all_txt=open(os.path.join(val_file_root,'test.txt'),'r')
  f_all_attr=open(os.path.join(val_file_root,'test.attr'),'r')#test.attr is the original attr
  f_neg_txt=open(os.path.join(val_file_root,'neg.txt'),'w')
  f_neg_attr=open(os.path.join(val_file_root,'neg.attr'),'w')
  f_pos_txt=open(os.path.join(val_file_root,'pos.txt'),'w')
  f_pos_attr=open(os.path.join(val_file_root,'pos.attr'),'w')
  txt_lines=f_all_txt.readlines()
  attr_lines=f_all_attr.readlines()
  counts=len(txt_lines)
 
  for txt,attr in zip(txt_lines,attr_lines):
        if attr[0]=='n':#here will flip,original is negative,so the attr should be positive
            f_pos_txt.write(txt)
            f_pos_attr.write(attr)
        else: 
            f_neg_txt.write(txt)
            f_neg_attr.write(attr)
  f_all_txt.close()
  f_all_attr.close()
  f_neg_txt.close()
  f_neg_attr.close()
  f_pos_txt.close()
  f_pos_attr.close()

  
  val_0=os.path.join(val_file_root,"neg.txt")
  val_1=os.path.join(val_file_root,"pos.txt")
  val_0_attr=os.path.join(val_file_root,"neg.attr")
  val_1_attr=os.path.join(val_file_root,"neg.attr")
  ppl_0=subprocess.getoutput(
    "./scripts/yelp/eval_lm.sh 0 pretrained_lm/{2}_style0/model.pt {0} {1}".format(val_0,val_0_attr,dataset)
  )
  log_string = "\nLM test_0\n"
  log_string +="\n{}".format(ppl_0)
  log_string +="\nppl test_0 finish"
  print(log_string)
  print("now is for check write fiel",val_0,val_1)
  ppl_1=subprocess.getoutput(
    "./scripts/yelp/eval_lm.sh 1 pretrained_lm/{2}_style1/model.pt {0} {1}".format(val_1,val_1_attr,dataset)
  )
  log_string = "\nLM test_1\n"
  # ppl_all = ppl_1.split('\n')
  # ppl_str=ppl_all[-3]
  # #deal with output 
  # print('\n')
  # print(ppl_str)
  # reg_ppl=re.compile("VAL")
  # try:
  #   ppl_t = reg_ppl.match(ppl_all).group(1)
  # except:
  #   ppl_t= "not find!"
  log_string +="\n{}".format(ppl_1)
  # log_string +="\n{} ".format(ppl_str)
  # log_string +="\n{} ".format(ppl_t)
  log_string +="\nppl test_1 finish"
  print(log_string)

  ppl=(ppl_0,ppl_1)

  #cls
  val_all=valid_hyp_file
  val_trg_all=os.path.join(val_file_root,"test.attr")
  cls = subprocess.getoutput(
    "./scripts/yelp/yelp_classify_test.sh {0} {1} {2}".format(val_all,val_trg_all,dataset)
  )
  log_string ="test_cls_acc"
  log_string +="\n{}".format(cls)
  cls_str = cls.split('\n')[-1].strip()
  log_string +="\n{}".format(cls_str)
  log_string +="\ncls_acc test finish"

  print(log_string)

  return valid_bleu,ppl,cls


def get_performance(crit, trans_logits, noise_logits, labels, hparams, x_len, sum_loss=True):
  # average over length
  x_len_t = torch.tensor(x_len, dtype=torch.float, requires_grad=False, device=hparams.device)
  x_len_t = x_len_t - 1
  batch_size = len(x_len)
  mask = (labels == hparams.pad_id)
  if hparams.bt:
    trans_logits = trans_logits.view(-1, hparams.src_vocab_size)
    trans_loss = crit(trans_logits, labels)
    trans_loss = trans_loss.view(batch_size, -1).sum(-1)
    _, trans_preds = torch.max(trans_logits, dim=1)
    trans_acc = torch.eq(trans_preds, labels).int().masked_fill_(mask, 0).sum().item()
  else:
    trans_loss = torch.zeros((batch_size), requires_grad=False, device=hparams.device)
    trans_acc = 0    

  if hparams.noise_flag:
    noise_logits = noise_logits.view(-1, hparams.src_vocab_size)
    noise_loss = crit(noise_logits, labels)
    noise_loss = noise_loss.view(batch_size, -1).sum(-1)
    _, preds = torch.max(noise_logits, dim=1)
    acc = torch.eq(preds, labels).int().masked_fill_(mask, 0).sum().item()
  else:
    noise_loss = torch.zeros((batch_size), requires_grad=False, device=hparams.device)
    acc = 0

  if hparams.avg_len:
    noise_loss = noise_loss / x_len_t
    trans_loss = trans_loss / x_len_t

  trans_loss = trans_loss.sum()
  noise_loss = noise_loss.sum()
  loss = trans_loss + hparams.noise_weight * noise_loss
  #loss = noise_loss.sum()
  return loss, trans_loss, noise_loss, acc, trans_acc

def count_params(params):
  num_params = sum(p.data.nelement() for p in params)
  return num_params

def save_checkpoint(extra, model, optimizer, hparams, path):
  print("Saving model to '{0}'".format(path))
  torch.save(extra, os.path.join(path, "extra.pt"))
  torch.save(model, os.path.join(path, "model.pt"))
  torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
  torch.save(hparams, os.path.join(path, "hparams.pt"))
  torch.save(model.state_dict(), os.path.join(path, "model.dict"))

class Logger(object):
  def __init__(self, output_file):
    self.terminal = sys.stdout
    self.log = open(output_file, "a")

  def write(self, message):
    print(message, end="", file=self.terminal, flush=True)
    print(message, end="", file=self.log, flush=True)

  def flush(self):
    self.terminal.flush()
    self.log.flush()

def set_lr(optim, lr):
  for param_group in optim.param_groups:
    param_group["lr"] = lr

def init_param(p, init_type="uniform", init_range=None):
  if init_type == "xavier_normal":
    init.xavier_normal(p)
  elif init_type == "xavier_uniform":
    init.xavier_uniform(p)
  elif init_type == "kaiming_normal":
    init.kaiming_normal(p)
  elif init_type == "kaiming_uniform":
    init.kaiming_uniform(p)
  elif init_type == "uniform":
    #assert init_range is not None and init_range > 0
    init.uniform_(p, -init_range, init_range)
  else:
    raise ValueError("Unknown init_type '{0}'".format(init_type))


def get_attn_subsequent_mask(seq, pad_id=0):
  """ Get an attention mask to avoid using the subsequent info."""

  assert seq.dim() == 2
  batch_size, max_len = seq.size()
  sub_mask = torch.triu(
    torch.ones(max_len, max_len), diagonal=1).unsqueeze(0).repeat(
      batch_size, 1, 1).type(torch.ByteTensor)
  if seq.is_cuda:
    sub_mask = sub_mask.cuda()
  return sub_mask

def grad_clip(params, grad_bound=None):
  """Clipping gradients at L-2 norm grad_bound. Returns the L-2 norm."""

  params = list(filter(lambda p: p.grad is not None, params))
  total_norm = 0
  for p in params:
    if p.grad is None:
      continue
    param_norm = p.grad.data.norm(2)
    total_norm += param_norm ** 2
  total_norm = total_norm ** 0.5

  if grad_bound is not None:
    clip_coef = grad_bound / (total_norm + 1e-6)
    if clip_coef < 1:
      for p in params:
        p.grad.data.mul_(clip_coef)
  return total_norm

