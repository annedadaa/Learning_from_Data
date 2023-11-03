#!/usr/bin/env python

""" Exploring binary text classification task using BERT, RoBERTa, DistilBERT, ..."""

# Import all necessary libraries.
import pandas as pd
import numpy as np
import random
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score
import emoji
from wordsegment import load, segment


def create_arg_parser():
    parser = argparse.ArgumentParser()
    # Filepath arguments
    parser.add_argument("-tr", "--train_file", default="preprocessed_data/train.tsv", type=str,
                        help="Input file to learn from (default train.tsv)",)
    
    parser.add_argument("-d", "--dev_file", type=str, default="preprocessed_data/dev.tsv",
                        help="Separate dev set to read in (default dev.tsv)",)
    
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set",)
    
    parser.add_argument("-m", "--model", type=str, default="bert-base-cased",
                        help="A language model model to finetune",)
    
    # Preprocessing arguments

    parser.add_argument("-p", "--prep", action="store_true",
                        help="Whether to preprocess the data (it's time consuming)",)

    # Training arguments

    parser.add_argument("--epochs", default=3, type=int, help="Set number of epochs")
    parser.add_argument("--lr", default=5e-5, type=float, help="Set learning rate")

    args = parser.parse_args()
    return args


class Data(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item
    
    def __len__(self):
        return len(self.labels)


def read_corpus(corpus_file):
    """
    Function that reads the corpus and gets texts and labels from it.
    Args:
        corpus_file (str): Path to corpus file.
        preprocessed (bool): Whether tex

    Returns:
        documents (List[str]), labels (List[str]): 2 lists of texts and labels respectively.
    """
    documents = []
    labels = []

    corpus = pd.read_table(corpus_file, names=['text', 'label'], header=None)
    
    documents = corpus['text'].to_list()
    labels = corpus['label'].to_list()

    return documents, labels


def preprocessing(documents):
    """
    Function to clean and preprocess texts.
    Args:
        documents (List[str]): list of texts.

    Returns:
        cleaned_hash_docs (List[str]): list of cleaned and preprocessed texts.
    """
    cleaned_user_docs = [doc.replace('@USER', '') for doc in documents]
    cleaned_url_docs = [doc.replace('URL', 'http') for doc in cleaned_user_docs]
    cleaned_emoji_docs = [emoji.demojize(doc) for doc in cleaned_url_docs]

    cleaned_hash_docs = []
    for doc in cleaned_emoji_docs:
        cleaned_doc = ''
        for word in doc.split():
            if word.startswith("#"):
                word = word.replace("#", "")
                load()
                word_list = segment(word)
                word = ' '.join(word_list)
            cleaned_doc = cleaned_doc + ' ' + word
        cleaned_doc = " ".join(cleaned_doc.split())
        cleaned_hash_docs.append(cleaned_doc)

    return cleaned_hash_docs


def seed_all(seed_value):
    """
    Function to set seed.
    Args:
        seed_value (int): seed value.

    Returns:
        None
    """

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def get_tokens(tokenizer, documents):
    """
    Function to tokenize texts.
    Args:
        tokenizer: model tokenizer.
        documents (List[str]): list of texts.

    Returns:
        tokens (dict): embedded tokens.
    """

    tokens = tokenizer.batch_encode_plus(
        documents,
        max_length = 64,
        padding = 'max_length',
        truncation = True)

    return tokens


def compute_metrics(pred):
    """
    Compute F1 macro score for evaluation when training.
    Args:
        pred: predictions by model.

    Returns:
        {'F1': f1} (dict): computed F1 macro score.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='macro')
    return {'F1': f1}


def train(model, tokenizer, train_dataset, dev_dataset, n_epoch, lr):
    """
    Train the model.
    Args:
        model: model to train.
        tokenizer: tokenizer.
        train_dataset (dict): train set.
        dev_dataset (dict): development set.
        n_epoch (int): number of epochs.
        lr (float): learning rate.

    Returns:
        trainer: trained model.
    """

    training_args = TrainingArguments(
        output_dir = './train_clf/results',
        num_train_epochs = n_epoch,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 32,
        weight_decay =0.01,
        logging_dir = './train_clf/logs',
        load_best_model_at_end = True,
        learning_rate = lr,
        evaluation_strategy ='epoch',
        logging_strategy = 'epoch',
        save_strategy = 'epoch',
        save_total_limit = 1,
        seed=42)

    trainer = Trainer(model=model,
                    tokenizer = tokenizer,
                    args = training_args,
                    train_dataset = train_dataset,
                    eval_dataset = dev_dataset,
                    compute_metrics = compute_metrics)

    trainer.train()

    return trainer


def get_prediction(trainer, test_documents):
    """
    Make predictions on given set of texts.
    Args:
        trainer: model to make predictions.
        test_documents (List[str]): list of texts.

    Returns:
        labels (List[int]): list of predicted values.
    """
    test_pred = trainer.predict(test_documents)
    labels = np.argmax(test_pred.predictions, axis = -1)

    return labels


if __name__ == "__main__":

    seed_all(42)
    args = create_arg_parser()

    # Mapping labels
    label_to_idx = {'NOT': 0, 'OFF': 1}
    idx_to_label = {0: 'NOT', 1: 'OFF'}

    train_documents, Y_train = read_corpus(args.train_file)
    dev_documents, Y_dev = read_corpus(args.dev_file)

    # If you want to run code using original data. We already preprocessed files
    # because preprocessing is time-consuming.
    if args.prep:

        train_documents = preprocessing(train_documents)
        dev_documents = preprocessing(dev_documents)

    # Turn string labels into numbers
    Y_train = [label_to_idx[i] for i in Y_train]
    Y_dev = [label_to_idx[i] for i in Y_dev]

    # Model initialization. Data tokenization.
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2,
                                                               ignore_mismatched_sizes=True).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_tokens = get_tokens(tokenizer, train_documents)
    dev_tokens = get_tokens(tokenizer, dev_documents)

    train_dataset = Data(train_tokens, Y_train)
    dev_dataset = Data(dev_tokens, Y_dev)

    trainer = train(model, tokenizer, train_dataset, dev_dataset, args.epochs, args.lr)

    # If test file specified, predictions will be calculated.
    if args.test_file:
        test_documents, Y_test = read_corpus(args.test_file)

        if args.prep:
            test_documents = preprocessing(test_documents)

        Y_test = [label_to_idx[i] for i in Y_test]
        
        test_tokens = get_tokens(tokenizer, test_documents)
        test_dataset = Data(test_tokens, Y_test)

        y_preds = get_prediction(trainer, test_dataset)

        # Saving output
        data = {'text': test_documents, 'label': [idx_to_label[i] for i in y_preds]} 
        df = pd.DataFrame(data)

        model_name = args.model
        if '/' in model_name:
            model_name = model_name.split('/')[1]
        
        df.to_csv('output/{}_preds.tsv'.format(model_name), sep="\t", header=False) 
