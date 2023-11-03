### Offensive Language Identification

#### To get started

Clone this repository

```
git clone https://github.com/annedadaa/Offensive_Language_Identification.git
```

You should run the all scripts from the root directory
```
cd Offensive_Language_Identification
```

To train models:
```
python3 evaluate/evaluate.py --pred output/SVM/svm_preds.tsv --cf
```

#### Models evaluation
 
To evaluate different models using saved outputs use the command lines bellow. 
Parameters:
- _--pred_: path to saved output file with predictions
- _--cf_ : creates confusion matrix

SVM:
```
python3 evaluate/evaluate.py --pred output/SVM/svm_preds.tsv --cf
```
LSTM:
```
python3 evaluate/evaluate.py --pred output/LSTM/lstm_preds.tsv --cf
```
RoBERTa:
```
python3 evaluate/evaluate.py --pred output/LMs/twitter-roberta-base-sentiment-latest_preds.tsv --cf
```
