#!/usr/bin/env python

"""Models evaluation."""

# Import all necessary libraries.
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def create_arg_parser():
    parser = argparse.ArgumentParser()
    # Filepath arguments
    parser.add_argument("-t", "--test_file", default="preprocessed_data/test.tsv", type=str,
                        help="If added, use trained model to predict on test set",)
    
    parser.add_argument("-p", "--preds_file", default="output/test.tsv", type=str,
                        help="Input file to learn from (default train.tsv)",)


    # Argument to call confusion matrix calculation
    parser.add_argument("--cf", action="store_true",
                        help="Whether to calculate confusion matrix for predictions",)

    args = parser.parse_args()
    return args


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


def get_confusion_matrix(y_test, y_pred, classes):
    """
    Make predictions and measure accuracy on our own test set (that we split off train).
    Args:
        X_test (list): test documents.
        y_test (list): test labels.
        y_pred (list): predicted labels.

    Returns:
        None
    """

    confusion_matrix_result = confusion_matrix(y_test, y_pred)

    # plot confusion matrix
    ax = plt.subplot()

    # annot=True to annotate cells, ftm='g' to disable scientific notation
    sn.heatmap(confusion_matrix_result, annot=True, fmt="g", ax=ax)

    # labels, title and ticks
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    ax.figure.savefig("LM_confusion_matrix.png")


if __name__ == "__main__":

    args = create_arg_parser()

    _, Y_test = read_corpus(args.test_file)
    _, Y_pred = read_corpus(args.preds_file)

    if args.cf:
        classes = ['NOT', 'OFF']
        get_confusion_matrix(Y_test, Y_pred, classes)

    test_classification_report_results = classification_report(Y_test, Y_pred)
    print(test_classification_report_results)

    f1 = f1_score(Y_test, Y_pred, average='macro')
    print('F1', f1)
