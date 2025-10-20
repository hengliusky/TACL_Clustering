## Title
Topology-Aware Contrastive Learning for Attributed Graph Clustering

### Training
For example, an example run on the CORA dataset is:
```
python3 train.py -dataset=cora
```
## Results
The visualization clustering comparisons on the CORA, and AMAP of our proposed approach with
Convert, CCGC and GraphLearner.
### Cora 
<p align="center">
  <img width="960" height="190" src="/results/image1.png"/>
</p>

### Citeseer 
<p align="center">
  <img width="960" height="190" src="/results/image2.png"/>
</p>


## Requirements
The code is built with:

* torch==1.8.0
* tqdm==4.61.2
* numpy==1.21.0
* munkres==1.1.4
* scikit_learn==1.0


## Datasets(https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering/tree/main/dataset)

### CORA:
The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links.

### ACM:
This is a paper network from the ACM dataset. There is an edge between two papers if they are written by same author. Paper features are the bag-of-words of the keywords. We select papers published in KDD, SIGMOD, SIGCOMM, MobiCOMM and divide the papers into three classes (database, wireless communication, data mining) by their research area.

### AMAP:
A-Computers and A-Photo are extracted from Amazon co-purchase graph, where nodes represent products, edges represent whether two products are frequently co-purchased or not, features represent product reviews encoded by bag-of-words, and labels are predefined product categories.

### BAT:
Data collected from the National Civil Aviation Agency (ANAC) from January to December 2016. It has 131 nodes, 1,038 edges (diameter is 5). Airport activity is measured by the total number of landings plus takeoffs in the corresponding year.

### EAT:
Data collected from the Statistical Office of the European Union (Eurostat) from January to November 2016. It has 399 nodes, 5,995 edges (diameter is 5). Airport activity is measured by the total number of landings plus takeoffs in the corresponding period.

### UAT:
Data collected from the Bureau of Transportation Statistics from January to October, 2016. It has 1,190 nodes, 13,599 edges (diameter is 8). Airport activity is measured by the total number of people that passed (arrived plus departed) the airport in the corresponding period.