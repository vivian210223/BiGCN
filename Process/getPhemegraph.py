# -*- coding: utf-8 -*-
import os
import pdb
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
cwd=os.getcwd()
class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        self.parent = None

def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    for pair in Str.split(' '):
        freq=float(pair.split(':')[1])
        index=int(pair.split(':')[0])
        if index<=5000:
            wordFreq.append(freq)
            wordIndex.append(index)
    return wordFreq, wordIndex

def constructMat(tree):
    index2node = {}
    print(tree)
    count=0
    for i in tree:
        #print(i)
        if i >count:
          count=i
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        ## not root node ##
        if not indexP == 'None':
            #  print(indexP)
            put=int(indexP)
            if put not in index2node.keys():
              put=1
            nodeP = index2node[put]   
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            rootindex=indexC-1
            root_index=nodeC.index
            root_word=nodeC.word
    rootfeat = np.zeros([1, 5000])
    if len(root_index)>0:
        rootfeat[0, np.array(root_index)] = np.array(root_word)
    matrix=np.zeros([count,count])
    row=[]
    col=[]
    x_word=[]
    x_index=[]
    for index_i in index2node.keys():
        for index_j in index2node.keys():
            if index2node[index_i].children != None and index2node[index_j] in index2node[index_i].children:
                matrix[index_i-1][index_j-1]=1
                row.append(index_i-1)
                col.append(index_j-1)
        x_word.append(index2node[index_i].word)
        x_index.append(index2node[index_i].index)
    edgematrix=[row,col]
    return x_word, x_index, edgematrix,rootfeat,rootindex

def getfeature(x_word,x_index):
    x = np.zeros([len(x_index), 5000])
    for i in range(len(x_index)):
        if len(x_index[i])>0:
            x[i, np.array(x_index[i])] = np.array(x_word[i])
    return x

def main(obj):
    treePath=''
    labelPath=''
    if 'twitter_15' in obj:
        treePath = '/mnt/ssd1/vivianweng/twitter/data.TD_RvNN.vol_5000.txt'
        labelPath = '/mnt/ssd1/vivianweng/twitter/data.label.txt'
    elif 'twitter_16' in obj:
        treePath = '/mnt/ssd1/vivianweng/twitter/twitter_16/data.TD_RvNN.vol_5000.txt'
        labelPath = '/mnt/ssd1/vivianweng/twitter/twitter_16/data.label.txt'
    elif 'pheme' in obj:
        treePath = '/mnt/ssd0/yunzhu/AgainstRumor/data/Pheme/data.TD_RvNN.vol_5000.txt'
        labelPath = '/mnt/ssd1/vivianweng/twitter/pheme/data.label.txt'
    
    print("reading twitter tree")
    treeDic = {}
    for line in open(treePath):
        # maxL: max # of clildren nodes for a node; max_degree: max # of the tree depth
        line = line.rstrip()
        #print(line)
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]

        #print('eid: ', eid)
        #print('indexP: ', indexP)
        #print('indexC: ', indexC)
        #print('max_degree: ', max_degree)
        #print('maxL: ', maxL)
        #print('Vec: ', Vec)
        if not treeDic.__contains__(eid):
            # If the event id hasn't been contained
            treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
    print('tree no:', len(treeDic))
    
    labelset_nonR, labelset_R = ['non-rumours','non-rumor'], ['rumours','rumor']

    print("loading tree label")
    event, y = [], []
    l1 = l2 = 0
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        label,eid=[],[]
        if 'twitter' in obj:
          label, eid = line.split('\t')[0], line.split('\t')[1]
        elif 'pheme' in obj:
          label, eid = line.split('\t')[0], line.split('\t')[2]
        label=label.lower()
        event.append(eid)
        if label in labelset_nonR:
            labelDic[eid]=0
            l1 += 1
        if label  in labelset_R:
            labelDic[eid]=1
            l2 += 1
    print(len(labelDic))
    print(l1, l2)

    def loadEid(event,id,y):
        if event is None:
            return None
        if len(event) < 2:
            return None
        if len(event)>= 2:
            x_word, x_index, tree, rootfeat, rootindex = constructMat(event) 
            x_x = getfeature(x_word, x_index) # x_word: the occur times of words, x_index: the index of words
            rootfeat, tree, x_x, rootindex, y = np.array(rootfeat), np.array(tree), np.array(x_x), np.array(
                rootindex), np.array(y)
            np.savez( os.path.join(cwd, 'data/'+obj+'graph/'+id+'.npz'), x=x_x,root=rootfeat,edgeindex=tree,rootindex=rootindex,y=y)
            return None
    print("loading dataset", )
    Parallel(n_jobs=30, backend='threading')(delayed(loadEid)(treeDic[eid] if eid in treeDic else None,eid,labelDic[eid]) for eid in tqdm(event))
    return

if __name__ == '__main__':
    obj= sys.argv[1]
    #obj = 'Twitter15'
    #obj = 'Pheme'
    
    path = os.path.join(cwd, 'data/'+obj+'graph/')
    if not os.path.exists(path):
        os.makedirs(path)
    main(obj)
