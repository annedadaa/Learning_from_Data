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

Install all the necessary libraries
```
pip install -r requirements.txt 
```

### Training

Scripts for training different models are in the _train_predict_ folder. To run (for example, Language Models) training script, use the command line below:
```
python3 train_predict/LanguageModels.py
```
Because Language Models are trained on GPU using pytorch, you have to specify CUDA IDs as follows:
```
CUDA_VISIBLE_DEVICES=1 python3 train_predict/LanguageModels.py
```
To save predictions on test set you should specify path to the test set:
```
CUDA_VISIBLE_DEVICES=1 python3 train_predict/LanguageModels.py --test_file dataset/preprocessed_data/test.tsv
```

For more detailed information on training please check [README file](https://github.com/annedadaa/Offensive_Language_Identification/blob/main/train_predict/README.md)  in the _train_predict_ folder.

### Evaluation
 
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
The best model checkpoint can be downloaded from [Google Drive](https://drive.google.com/file/d/1cxAcadm6C9MJpIErTCqaw5ISdKcvRkBu/view?usp=sharing).

For more detailed information on evaluation please check [README file](https://github.com/annedadaa/Offensive_Language_Identification/blob/main/evaluate/README.md) in the _evaluate_ folder.
