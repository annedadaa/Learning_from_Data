## Offensive Language Identification

### To get started

Clone this repository

```
git clone https://github.com/annedadaa/Offensive_Language_Identification.git
```

You should run the all scripts from the root directory
```
cd Offensive_Language_Identification
```

### Models training

Scripts for training different models are in the _train_predict_ folder. To run (for example, Language Models) training script, use the command line below:
```
python3 train_predict/LanguageModels.py
```
Because Language Models are trained on GPU, you have to specify CUDA IDs as follows:
```
CUDA_VISIBLE_DEVICES=1 python3 train_predict/LanguageModels.py
```
To save predictions on test set you should specify path to the test set:
```
CUDA_VISIBLE_DEVICES=1 python3 train_predict/LanguageModels.py --test_file dataset/preprocessed_data/test.tsv
```

For more detailed information on training please check README file in the _train_predict_ folder.

### Models evaluation
 
To evaluate different models using saved outputs use the command lines below:

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
