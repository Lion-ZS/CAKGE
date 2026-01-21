


## Requirements

All experiments are implemented on the NVIDIA RTX 3090Ti GPU with the PyTorch. The version of Python is 3.7.

Please run as follows to install all the dependencies:

```shell
pip3 install -r requirements.txt
```

## Usage

### Preparation

1. Unzip the dataset $data1.zip$ and $data2.zip$ in the folder of $data$.
2. Prepare the data processing in the shell:

```shell
python data_preprocess.py
```

### Main Results

3. Run the code with this in the shell:

```shell
python main.py -dataset ENTITY -gpu 2  
python main.py -dataset FACT -gpu 0
python main.py -dataset graph_equal -gpu 1
python main.py -dataset graph_higher -gpu 0
python main.py -dataset graph_lower -gpu 1
python main.py -dataset HYBRID -gpu 0
python main.py -dataset RELATION -gpu 1  
```

