   cd transductive && python -W ignore train.py --data_path=data/family --gpu 0
   cd transductive && python -W ignore train.py --data_path=data/umls --gpu 1
   cd transductive && python -W ignore train.py --data_path=data/WN18RR --gpu 2 
   cd transductive && python -W ignore train.py --data_path=data/fb15k-237 --gpu 3 
   cd transductive && python -W ignore train.py --data_path=data/nell --gpu 1
   cd transductive && python -W ignore train.py --data_path=data/yago3 --gpu 1