import os
from Process.dataset import GraphDataset,BiGraphDataset,UdGraphDataset
cwd=os.getcwd()


################################### load tree#####################################
def loadTree(dataname):
    treePath=''
    if 'twitter_15' in dataname:
        treePath = '/mnt/ssd1/vivianweng/twitter/data.TD_RvNN.vol_5000.txt'
    elif 'twitter_16' in dataname:
        treePath = '/mnt/ssd1/vivianweng/twitter/twitter_16/data.TD_RvNN.vol_5000.txt'
    elif 'pheme' in dataname:
        treePath = '/mnt/ssd0/yunzhu/AgainstRumor/data/Pheme/data.TD_RvNN.vol_5000.txt'
    print("reading twitter tree")
    treeDic = {}
    with open(treePath) as f:
      for line in f:
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
        if not treeDic.__contains__(eid):
          treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
      print('tree no:', len(treeDic))

    return treeDic

################################# load data ###################################
def loadData(dataname, treeDic,fold_x_train,fold_x_test,droprate):
    data_path=os.path.join(cwd, 'data', dataname+'graph')
    print("loading train set", )
    traindata_list = GraphDataset(fold_x_train, treeDic, droprate=droprate,data_path= data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = GraphDataset(fold_x_test, treeDic,data_path= data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadUdData(dataname, treeDic,fold_x_train,fold_x_test,droprate):
    data_path=os.path.join(cwd, 'data',dataname+'graph')
    print("loading train set", )
    traindata_list = UdGraphDataset(fold_x_train, treeDic, droprate=droprate,data_path= data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = UdGraphDataset(fold_x_test, treeDic,data_path= data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list

def loadBiData(dataname, treeDic, fold_x_train, fold_x_test, TDdroprate,BUdroprate):
    data_path = os.path.join(cwd,'data', dataname + 'graph')
    #print(data_path)
    print("loading train set", )
    #print(len(fold_x_train))
    traindata_list = BiGraphDataset(fold_x_train, treeDic, tddroprate=TDdroprate, budroprate=BUdroprate, data_path=data_path)
    print("train no:", len(traindata_list))
    print("loading test set", )
    testdata_list = BiGraphDataset(fold_x_test, treeDic, data_path=data_path)
    print("test no:", len(testdata_list))
    return traindata_list, testdata_list



