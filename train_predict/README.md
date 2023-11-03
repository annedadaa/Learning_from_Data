## Training different models

### Train Naive Bayes/Decision Tree/Random Forest/K-Nearest Neighbors/Support Vector Machine

Download spacy embeddings which are used for text lemmatization

```
python3 -m spacy download en_core_web_sm
```
Run the training script (for example using TD-IDF vectorizer and SVM model)
```
python3 train_predict/NB_DT_RT_KNN_SVM.py --model svm --vect tfidf
```

For all the optional command line arguments please see [the code](https://github.com/annedadaa/Offensive_Language_Identification/edit/main/train_predict/NB_DT_RT_KNN_SVM.py)

### Train LSTM
Download pretrained GloVe twitter embeddings and unzip archive

```
wget https://nlp.stanford.edu/data/glove.twitter.27B.zip
```
```
unzip glove.twitter.27B.zip
```
Run the training script
```
python3 train_predict/LSTM.py
```
**Additional command line arguments**

Paths:
- _train_file_: path to train set (default="dataset/preprocessed_data/train.tsv")
- _dev_file_: path to dev set (default="dataset/preprocessed_data/dev.tsv")
- _test_file_: path to test set
- _embeddings_: path to pretrained embeddings (default="glove.twitter.27B.200d.txt")
  
Architecture Parameters:
- _add_dense_: whether to add Dense layer between Embedding and LSTM layers
- _bilstm_: whether to use BiLSTM
  
Model Hyperparameters:
- _epochs_: number of epochs
- _lr_: learning rate
- _batch_size_: batch size
- _dropout_: dropout rate
- _optimizer_: model optimizer
- _hidden_size_: hidden size of (Bi)LSTM layers
- _maxlen_: maximum sequence length
  
Preprocessing Parameter:
- _prep_: whether to preprocess the data (it's time consuming)

### Train Pretrained Language Models
