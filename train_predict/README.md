Add more info about training models

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

### Train LSTM

### Train Pretrained Language Models
