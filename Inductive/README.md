python3 train.py -D fb237_v1 -T train -HW --gpu 0 -E reproduction
python3 train.py -D fb237_v2 -T train -HW --gpu 1 -E reproduction
python3 train.py -D fb237_v3 -T train -HW --gpu 0 -E reproduction
python3 train.py -D fb237_v4 -T train -HW --gpu 1 -E reproduction
python3 train.py -D WN18RR_v1 -T train -HW --gpu 0 -E reproduction
cd inductive  python  train.py --data_path=data/WN18RR_v2 --gpu 3
python3 train.py -D WN18RR_v3 -T train -HW --gpu 0 -E reproduction
python3 train.py -D WN18RR_v4 -T train -HW --gpu 1 -E reproduction
python3 train.py -D nell_v1 -T train -HW --gpu 0 -E reproduction
python3 train.py -D nell_v2 -T train -HW --gpu 0 -E reproduction
python3 train.py -D nell_v3 -T train -HW --gpu 0 -E reproduction
python3 train.py -D nell_v4 -T train -HW --gpu 0 -E reproduction