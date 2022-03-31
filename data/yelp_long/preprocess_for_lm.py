from os import POSIX_FADV_WILLNEED

from sklearn.decomposition import non_negative_factorization


a_file='dev_30.attr'
t_file='dev_30.txt'
pos_file='dev_30.pos'
neg_file='dev_30.neg'
pos_attr='dev_30_1.attr'
neg_attr='dev_30_0.attr'

a_train=open(a_file,'r')
t_train=open(t_file,'r')
pos_train=open(pos_file,'w')
neg_train=open(neg_file,'w')
pos_a=open(pos_attr,'w')
neg_a=open(neg_attr,'w')

alines=a_train.readlines()
tlines=t_train.readlines()

for a,t in zip(alines,tlines):
    if a[0] == "n":
        neg_train.write(t.strip()+"\n")
        neg_a.write(a.strip()+'\n')
    else:
        pos_train.write(t.strip()+'\n')
        pos_a.write(a.strip()+'\n')
print('finish dev set')



a_train.close()
t_train.close()
pos_train.close()
neg_train.close()
pos_a.close()
neg_a.close()


a_file='train_30.attr'
t_file='train_30.txt'
pos_file='train_30.pos'
neg_file='train_30.neg'
pos_attr='train_30_1.attr'
neg_attr='train_30_0.attr'

a_train=open(a_file,'r')
t_train=open(t_file,'r')
pos_train=open(pos_file,'w')
neg_train=open(neg_file,'w')
pos_a=open(pos_attr,'w')
neg_a=open(neg_attr,'w')

alines=a_train.readlines()
tlines=t_train.readlines()

for a,t in zip(alines,tlines):
    if a[0] == "n":
        neg_train.write(t.strip()+"\n")
        neg_a.write(a.strip()+'\n')
    else:
        pos_train.write(t.strip()+'\n')
        pos_a.write(a.strip()+'\n')
print('finish train set')



a_train.close()
t_train.close()
pos_train.close()
neg_train.close()
pos_a.close()
neg_a.close()
