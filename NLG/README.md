## NLG

### Setting up environment

    conda create -n pt python=3.7
    conda activate pt

    conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
    pip install pandas
    pip install matplotlib
    pip install nltk
    pip install bert_score


### Document explanation

- data

data storage

- result

generate result

- *.py


### how to implement 

1. data_process.py


    # processing and generate the data
    python data_process.py

2. nlg_train.py

choose  T5

    # training
    python nlt_train.py --mode t5


3. nlg_predict.py

Save the ouput in result folder

    # predict
    python nlt_predict.py --mode t5

4. nlg_metric.py


    # validate the output BLEU1-4, METEOR, and BERTscore
    python nlt_predict.py