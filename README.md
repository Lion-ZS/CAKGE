# CAKGE: Context-aware Adaptive Learning for Dynamic Knowledge Graph Embeddings

## Start
Requirements

- pytorch  1.9.1+cu111
- torch_scatter 2.0.8  
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html
- scipy 1.9.2
- prettytable
- networkx
- matplotlib

## For transductive reasoning

We can use the following commands to train the model and evaluate the link prediction performance of CAKGE on the WN18RR dataset under the transductive setting. If the program is interrupted, you can reduce the batchsize.

```
   cd transductive && python -W ignore train.py --data_path=data/family --gpu 0
   cd transductive && python -W ignore train.py --data_path=data/umls --gpu 1
   cd transductive && python -W ignore train.py --data_path=data/WN18RR --gpu 2 
   cd transductive && python -W ignore train.py --data_path=data/fb15k-237 --gpu 3 
   cd transductive && python -W ignore train.py --data_path=data/nell --gpu 1
   cd transductive && python -W ignore train.py --data_path=data/yago3 --gpu 1
```
 If the program stops running after a while, it may be due to out of memory usage. You can change the batchsize in the train.py.


For other datasets can refer to the corresponding folder.
   







