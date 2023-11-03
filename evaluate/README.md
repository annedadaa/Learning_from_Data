## Evaluation of different models

To evaluate different models using saved outputs use the command lines below:

SVM:
```
python3 evaluate/evaluate.py --pred output/SVM/svm_preds.tsv
```
LSTM:
```
python3 evaluate/evaluate.py --pred output/LSTM/lstm_preds.tsv
```
RoBERTa:
```
python3 evaluate/evaluate.py --pred output/LMs/twitter-roberta-base-sentiment-latest_preds.tsv
```

For all the required and optional command line arguments please see [evaluation code](https://github.com/annedadaa/Offensive_Language_Identification/blob/d1ded5ab60bbc45bef3a1312472380c1ef68848d/evaluate/evaluate.py#L13)
