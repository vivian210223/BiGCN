#! /bin/bash
#unzip -d ./data/Weibo ./data/Weibo/weibotree.txt.zip
#pip install -U torch==1.4.0 numpy==1.18.1
#pip install -r requirements.txt
#Generate graph data and store in /data/Weibograph
#python ./Process/getWeibograph.py
#Generate graph data and store in /data/Twitter15graph
#python ./Process/getPhemegraph.py pheme
#Reproduce the experimental results.
#python ./model/Weibo/BiGCN_Weibo.py 100
CUDA_VISIBLE_DEVICES=2 python ./model/Pheme/BiGCN_Pheme.py pheme 15
#python ./model/Twitter/BiGCN_Twitter.py Twitter16 100
