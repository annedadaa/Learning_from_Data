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

**Command line arguments:**

- --test_file: path to test file with labels
- --preds_file: path to output file with predictions
- --cf: calculate confusion matrix
