import random
from random import shuffle
import os


def load5foldData(obj):
    labelPath=''
    train='/train.label.txt'
    test='/test.label.txt'
    folder=['split_0','split_1','split_2','split_3','split_4']
    fold0_test,fold0_train,fold1_test,fold1_train,fold2_test,fold2_train,fold3_test,fold3_train,fold4_test,fold4_train =[],[],[],[],[],[],[],[],[],[]
    if 'twitter_15' in obj:
        labelPath = '/mnt/ssd1/vivianweng/twitter/'
    elif 'twitter_16' in obj:
        labelPath = '/mnt/ssd1/joshchang/twitter16/'
    elif 'pheme' in obj:
        labelPath = '/mnt/ssd0/yunzhu/AgainstRumor/data/Pheme/'
    
    counter=0
    for i in folder:
        path_train=labelPath+i+train
        path_test=labelPath+i+test
        with open(path_train) as f:
          for line in f:
              line = line.rstrip()
              label, eid = line.split('\t')[0], line.split('\t')[1]
              if('pheme' in obj):
                eid=line.split('\t')[-1]   
              
              if counter==0:
                fold0_train.append(eid)
              elif counter==1:
                fold1_train.append(eid)
              elif counter==2:
                fold2_train.append(eid)
              elif counter==3:
                fold3_train.append(eid)
              elif counter==4:
                fold4_train.append(eid)
        
        with open(path_test) as f:
          for line in f:
              line = line.rstrip()
              label, eid = line.split('\t')[0], line.split('\t')[1]
              if('pheme' in obj):
                eid=line.split('\t')[-1]
              if counter==0:
                fold0_test.append(eid)
              elif counter==1:
                fold1_test.append(eid)
              elif counter==2:
                fold2_test.append(eid)
              elif counter==3:
                fold3_test.append(eid)
              elif counter==4:
                fold4_test.append(eid)
        counter+=1       
      
    
    return list(fold0_test),list(fold0_train),\
           list(fold1_test),list(fold1_train),\
           list(fold2_test),list(fold2_train),\
           list(fold3_test),list(fold3_train),\
           list(fold4_test), list(fold4_train)
