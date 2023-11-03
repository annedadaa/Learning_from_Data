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

To get predictions on test set add _test_file_ argument
```
python3 train_predict/NB_DT_RT_KNN_SVM.py --model svm --vect tfidf --test_file dataset/preprocessed_data/test.tsv
```

For all the required and optional command line arguments please see [the code](https://github.com/annedadaa/Offensive_Language_Identification/blob/954ab945d65b8383fa3f9ecf654118f0793e71d0/train_predict/NB_DT_RT_KNN_SVM.py#L32)

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
To get predictions on test set add _test_file_ argument
```
python3 train_predict/LSTM.py --test_file dataset/preprocessed_data/test.tsv

```

For all the required and optional command line arguments please see [the code] (link)

### Train Pretrained Language Models
Run the training script on GPU (for example, with GPU ID=1)

```
CUDA_VISIBLE_DEVICES=1 python3 train_predict/LanguageModels.py
```
To get predictions on test set add _test_file_ argument

```
CUDA_VISIBLE_DEVICES=1 python3 train_predict/LanguageModels.py --test_file dataset/preprocessed_data/test.tsv
```
For all the required and optional command line arguments please see [the code](https://github.com/annedadaa/Offensive_Language_Identification/blob/59cfda15503a9b7ec0817d374525a41a8679495d/train_predict/LanguageModels.py#L18)

