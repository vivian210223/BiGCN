# BiGCN
```
run baseline model with our on process datasets
```
# install
```
pip install -U torch==1.4.0 numpy==1.18.1<br/>
pip install -r requirements.txt
```
# Generate graph data and store in /data/"datasetname"graph(ex:pheme,twitter_15,twitter_16)
```
python ./Process/getPhemegraph.py pheme<br/>
python ./Process/getPhemegraph.py twitter_15<br/>
python ./Process/getPhemegraph.py twitter_16<br/>
```
# Reproduce the experimental results.
python ./model/Pheme/BiGCN_Pheme.py "datasetname" "iteration time"<br/>
```
python ./model/Twitter/BiGCN_Twitter.py pheme 15<br/>
python ./model/Twitter/BiGCN_Twitter.py twitter_15 15<br/>
python ./model/Twitter/BiGCN_Twitter.py twitter_16 15<br/>
```
