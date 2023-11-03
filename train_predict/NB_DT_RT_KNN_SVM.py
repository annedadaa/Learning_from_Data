#!/usr/bin/env python

""" Exploring binary text classification task using 5 different machine learning algorithms."""

# Import all necessary libraries.
import numpy as np
import pandas as pd
import argparse
import spacy
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import random
import emoji
from wordsegment import load, segment

# Random seed to prevent actual randomness when reruning the code.
random.seed(42)
# Download list of english stopwords and spacy model.
spacy_model = spacy.load("en_core_web_sm")


# Add command line agrument parser and all needed options for it.
def create_arg_parser():
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("-tr", "--train_file", default="dataset/preprocessed_data/train.tsv", type=str,
                        help="Data file to train and evaluate the model (default train.tsv)",)
    
    parser.add_argument("-d", "--dev_file", type=str, default="dataset/preprocessed_data/dev.tsv",
                        help="Separate dev set to read in (default dev.tsv)",)
    
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set",)

    # comments for model hyperparams tuning and cross-validation
    parser.add_argument("-fp", "--find_params", action="store_true", 
                        help="Whether to run grid search",)

    # SVM hyperparams
    parser.add_argument("--C", default=1, type=float, 
                        help="Set C hyperparam for LinearSVC model (default 1)",)
    # NB hyperparams
    parser.add_argument("--alpha", default=1, type=float, 
                        help="Set alpha hyperparam for NaiveBayes model (default 1)",)
    
    parser.add_argument("--fit_prior", action="store_true", 
                        help="Set fit_prior hyperparam for NaiveBayes model (default False)",)
    
    # DT and RandomForestClassifier hyperparams
    parser.add_argument("--max_depth", default=None, type=int, 
                        help="Set max_depth hyperparam for DecisionTreeClassifier/RandomForestClassifier \
                        model (default None)",)
    
    parser.add_argument("--criterion", default="gini", choices=["gini", "entropy"],
                        help="Set criterion hyperparam for DecisionTreeClassifier/RandomForestClassifier \
                        model (default gini)",)
    
    parser.add_argument("--n_est", default=100, type=int,
                        help="Set n_estimators hyperparam for RandomForestClassifier model (default 100)",)
    
    # KNN hyperparams
    parser.add_argument("--n_neigh", default=5, type=int,
                        help="Set n_neighbors hyperparam for KNearestNeighborsClassifier model (default 5)",)
    
    parser.add_argument("--weights", default="uniform", choices=["uniform", "distance"],
                        help="Set weights hyperparam for KNearestNeighborsClassifier model (default uniform)",)

    # params for features setting (aka data processing) 
    parser.add_argument(
        "-l", "--lem", action="store_true", help="Lemmatize texts (default False)",)
    
    parser.add_argument("-ap", "--add_prep", action="store_true",
                        help="Whether to additionally preprocess tweets (it's time consuming)",)
    
    parser.add_argument("-ng", "--ngram", default="(1,3)",
                        type=tuple_type, help="Set ngram size for vectorizer (default (1,1)",)

    # Required argument
    requiredNamed = parser.add_argument_group("required arguments")

    requiredNamed.add_argument("-m", "--model", choices=["nb", "dt", "rf", "knn", "svm"],
                               help="Select the model for both classifications",
                               required=True,)
    
    requiredNamed.add_argument("-v", "--vect", choices=["tfidf", "cv"], 
                               help="Choose Vectorizer", required=True)


    args = parser.parse_args()
    return args


def tuple_type(string):
    """
    Function that converts string into tuple data type.
    Args:
        string (str): String that should be converted.
    Returns:
        tuple(mapped_int) (tuple(int)): tuple with integer values.
    """

    # Replace brackets by empty string, then convert string
    # to tuple variable.
    string = string.replace("(", "").replace(")", "")
    mapped_int = map(int, string.split(","))
    return tuple(mapped_int)


def read_corpus(corpus_file):
    """
    Function that reads the corpus and gets texts and labels from it.
    Args:
        corpus_file (str): Path to corpus file.

    Returns:
        documents (List[str]), labels (List[str]): 2 lists of texts and labels respectively.
    """
    documents = []
    labels = []

    corpus = pd.read_table(corpus_file, names=['text', 'label'], header=None)
    
    documents = corpus['text'].to_list()
    labels = corpus['label'].to_list()

    return documents, labels


def text_preprocessing(text, lemmatize=False, add_prep=False):
    """
    Function that processes the text (i.e., lemmatization, punctuation removal)
    Args:
        word_tokens (List[str]): List of words
        lemmatize (bool): Whether to lemmatize words from text

    Returns:
        words.split() (List[str]): list of (preprocesses) words.
    """

    # If none of the preprocessing steps is called, this variable is returned.
    # Otherwise, it's overwriten and returned.
    words = text

    # Lemmatization of text using SpaCy
    if lemmatize:
        doc = spacy_model(words)
        lemmas_list = [token.lemma_ for token in doc]
        words = " ".join(lemmas_list)

    # Additional preprocessing cleaning step which is used by default in this task.
    # However, it's time consuming. 
    # For this purpose we created preprocessed files so we don't call this argument each time.
    if add_prep:
        words = words.replace('@USER', '')
        words = words.replace('URL', 'http')
        words = emoji.demojize(words)

        clean_text = ''
        for word in words.split():
            if word.startswith("#"):
                word = word.replace("#", "")
                load()
                word_list = segment(word)
                word = ' '.join(word_list)
            clean_text = clean_text + ' ' + word
        clean_text = " ".join(clean_text.split())
        
        words = clean_text

    return words.split()


def grid_search(params, model, X_train, y_train, model_name):
    """
    Function to search for the best hyperparameters for the selected model.
    Args:
        params (dict): Model hyperparameters.
        model (pipeline): Pipeline with vectorizer and model combined.
        X_train (List[str]): Train texts.
        y_train (List[str]): Train labels.
    Returns:
        grid_search.best_estimator_ (estimator): Estimator with the best hyperparameters.
    """
    grid_search = GridSearchCV(model, params, cv=5, verbose=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("Best parameters for the {} model are:".format(model_name), grid_search.best_estimator_)
    return grid_search.best_estimator_


def identity(inp):
    """
    Dummy function that just returns the input.
    Args:
        inp (Any): any input.
    Returns:
        inp (Any): any output the same as input.
    """
    return inp


if __name__ == "__main__":
    args = create_arg_parser()

    # Read the original dataset
    train_documents, Y_train = read_corpus(args.train_file)
    dev_documents, Y_dev = read_corpus(args.dev_file)

    # Preprocess data if any preprocessing steps are specified in the command line.
    X_train = [
        text_preprocessing(text, lemmatize=args.lem, add_prep=args.add_prep)
        for text in train_documents]
    X_dev = [
        text_preprocessing(text, lemmatize=args.lem, add_prep=args.add_prep)
        for text in dev_documents]

    # Split the data to train/test sets.
    # X_train, X_test, y_train, y_test = split_data(clean_documents, labels)

    if args.vect == "tfidf":
        # TfIdf vectorizer with chosen n-grams specified in the command line.
        vec = TfidfVectorizer(
            preprocessor=identity, tokenizer=identity, ngram_range=args.ngram
        )
    elif args.vect == "cv":
        # Bag of Words vectorizer with chosen n-grams specified in the command line.
        vec = CountVectorizer(
            preprocessor=identity, tokenizer=identity, ngram_range=args.ngram
        )

    # Handle 5 different classifiers and their params based on the arguments from the command line.
    # Each one then is combined with the chosen vectorizer.
    if args.model == "nb":
        classifier = Pipeline([("vec", vec), ("cls", MultinomialNB(alpha=args.alpha))])

    elif args.model == "dt":
        classifier = Pipeline([("vec", vec), 
                               ("cls", DecisionTreeClassifier(criterion=args.criterion, max_depth=args.max_depth),),])
    elif args.model == "rf":
        classifier = Pipeline([("vec", vec), 
                               ("cls", RandomForestClassifier(n_estimators=args.n_est, 
                                                              criterion=args.criterion, 
                                                              max_depth=args.max_depth,),),])
        
    elif args.model == "knn":
        classifier = Pipeline([("vec", vec), 
                               ("cls", KNeighborsClassifier(n_neighbors=args.n_neigh, 
                                                            weights=args.weights),),])

    elif args.model == "svm":
        classifier = Pipeline([("vec", vec), ("cls", LinearSVC(C=args.C))])

    # Apply gridsearch if it's called from the command line.
    if args.find_params:
        # FOR MultinomialNB
        if args.model == "nb":
            param_grid = {"cls__alpha": (0.01, 0.5, 1), "cls__fit_prior": (True, False)}
        # FOR DecisionTreeClassifier
        if args.model == "dt":
            param_grid = {
                "cls__criterion": ["gini", "entropy"],
                "cls__max_depth": np.arange(0, 10),
            }
        # FOR RandomForestClassifier
        if args.model == "rf":
            param_grid = {
                "cls__n_estimators": [100, 200, 500],
                "cls__criterion": ["gini", "entropy"],
                "cls__max_depth": np.arange(0, 10),
            }
        # FOR KNeighborsClassifier
        if args.model == "knn":
            param_grid = {
                "cls__n_neighbors": range(1, 15),
                "cls__weights": ["uniform", "distance"],
            }
        # FOR LinearSVM
        if args.model == "svm":
            param_grid = {"cls__C": [0.01, 0.1, 0, 1, 5, 10]}

        # define the model which will be used
        best_model = grid_search(param_grid, classifier, X_train, Y_train, args.model)
    else:
        best_model = classifier

    # Fit the model to train and then predict on the test set
    best_model.fit(X_train, Y_train)
    Y_pred = best_model.predict(X_dev)

    # Output of results: classification report with precision/recall/f1 score
    # for each class and confusion matrix
    f1 = f1_score(Y_dev, Y_pred, average='macro')
    print(f1)

    # Do predictions on specified test set
    if args.test_file:
        # Read in test set and vectorize
        test_documents, Y_test = read_corpus(args.test_file)
        X_test = [
        text_preprocessing(text, lemmatize=args.lem, add_prep=args.add_prep)
        for text in test_documents]

        Y_pred = best_model.predict(X_test)

        # Saving output
        data = {'text': test_documents, 'label': Y_pred} 
        df = pd.DataFrame(data)

        model_name = args.model
        if '/' in model_name:
            model_name = model_name.split('/')[1]
        
        df.to_csv('output/SVM/{}_preds.tsv'.format(args.model), sep="\t", header=False) 
