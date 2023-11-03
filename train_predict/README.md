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

Additional Parameters:
-
-
-

### Train LSTM
Download pretrained GloVe twitter embeddings and unzip archive

```
wget https://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
```
Run the training script
```
python3 train_predict/LSTM.py --model svm --vect tfidf
```
Additional Parameters:
- --train_file: path to train set (default="dataset/preprocessed_data/train.tsv")
- --dev_file: path to dev set (default="dataset/preprocessed_data/dev.tsv")
- --test_file: path to test set
- --embeddings: path to pretrained embeddings (default="glove.twitter.27B.200d.txt")
- 
### Train Pretrained Language Models
