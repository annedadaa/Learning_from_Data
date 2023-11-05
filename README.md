## Offensive Language Identification
### Final Metrics

Model | Dev F1 macro | Test F1 macro
--- | --- | --- 
SVM | 68.7 | 67.5
BiLSTM | 71.1 | 70.9
RoBERTa | 79.5 | **82.1**
RoBERTa + add. data | 93.9 | 73.7

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

For more detailed information on evaluation please check [README file](https://github.com/annedadaa/Offensive_Language_Identification/blob/main/evaluate/README.md) in the _evaluate_ folder.

### Training the best model on additional data

As the additional dataset to finetune the best model, we used [Toxic Tweets Dataset](https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset/data). Train and development parts of additional dataset can be found in the _dataset_ folder. Download the best model checkpoint from [Google Drive](https://drive.google.com/file/d/1cxAcadm6C9MJpIErTCqaw5ISdKcvRkBu/view?usp=sharing) and unzip.

Training the model on additional data:
```
CUDA_VISIBLE_DEVICES=1 python3 train_predict/LanguageModels.py --train_file dataset/additional_data/train.tsv --dev_file dataset/additional_data/dev.tsv --model checkpoint-roberta-best
```
For more detailed information on training please check [README file](https://github.com/annedadaa/Offensive_Language_Identification/blob/main/train_predict/README.md)  in the _train_predict_ folder.

### Detoxification

To detoxify the dataset, use the following command line:
```
CUDA_VISIBLE_DEVICES=1 python3 detoxification/detoxify.py --detox
```
To compute final table reporting the results of detoxification, run the following:
```
CUDA_VISIBLE_DEVICES=1 python3 detoxification/detoxify.py
```
For more detailed information on detoxification please check [README file]([https://github.com/annedadaa/Offensive_Language_Identification/blob/main/train_predict/README.md](https://github.com/annedadaa/Offensive_Language_Identification/tree/main/detoxification)https://github.com/annedadaa/Offensive_Language_Identification/tree/main/detoxification/README.md) in the _detoxification_ folder.
