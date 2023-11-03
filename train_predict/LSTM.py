#!/usr/bin/env python

"""
Exploring binary text classification task using LSTM.
"""

import random as python_random
import argparse
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from sklearn.metrics import f1_score
from keras.initializers import Constant
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
import emoji
from wordsegment import load, segment

# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    # Filepath arguments
    parser.add_argument("-tr", "--train_file", default="dataset/preprocessed_data/train.tsv", type=str,
                        help="Input file to learn from (default train.tsv)",)
    
    parser.add_argument("-d", "--dev_file", type=str, default="dataset/preprocessed_data/dev.tsv",
                        help="Separate dev set to read in (default dev.tsv)",)
    
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set",)
    
    parser.add_argument("-e", "--embeddings", default="glove.twitter.27B.200d.txt", type=str,
                        help="Embedding file we are using (default glove.twitter.27B.200d.txt)",)

    # Layers arguments
    parser.add_argument("--add_dense", action="store_true", 
                        help="Whether to add Dense layer between Embedding and LSTM layers.",)
    
    parser.add_argument("--bilstm", action="store_true", help="Whether to use Bidirectional LSTM.")

    # Hyperparameters arguments
    parser.add_argument("--epochs", default=30, type=int, help="Set number of epochs")
    parser.add_argument("--lr", default=0.001, type=float, help="Set learning rate")
    parser.add_argument("--batch_size", default=32, type=int, help="Set batch size")
    parser.add_argument("--dropout", default=0.2, type=float, help="Set dropout rate")
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "SGD"], 
                        help="Set optimizer (default Adam)",)
    parser.add_argument("--hidden_size", default=150, type=int, help="Set hidden size")
    parser.add_argument("--maxlen", default=50, type=int, help="Set max length of sequence")
    
    # Preprocessing arguments

    parser.add_argument("-p", "--prep", action="store_true",
                        help="Whether to preprocess the data (it's time consuming)",)

    args = parser.parse_args()
    return args


def read_corpus(corpus_file, preprocessed=True):
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

    if preprocessed:
      corpus = pd.read_table(corpus_file, names=['text', 'label'], header=None)
    else:
      corpus = pd.read_table(corpus_file, names=['text', 'label'], header=None)

    documents = corpus['text'].to_list()
    labels = corpus['label'].to_list()

    return documents, labels


def read_embeddings(embeddings_file):
    """
    Function that reads embeddings from the file.
    Args:
        embeddings_file (str): Path to embedding file.

    Returns:
        (dict): dictionary with words and their corresponding embeddings.
    """
    embedding_dict = {}
    with open(embeddings_file, 'r') as glove:
      for line in glove:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], 'float32')
        embedding_dict[word] = vectors

    return embedding_dict


def preprocessing(documents):
    """
    Function that clean and preprocess texts.
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


def get_emb_matrix(voc, emb):
    """
    Function that gets embedding matrix given vocab and the embeddings
    Args:
        voc (list): vocabulary.
        emb (dict): dictionary with words and their corresponding embeddings.

    Returns:
        embedding_matrix (np.array): matrix with pretrained embeddings.
    """

    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def create_model(
    Y_train,
    emb_matrix,
    learning_rate,
    dropout,
    optimizer,
    additional_dense,
    bidirectional,
    hidden_size
):
    """
    Function that creates the LSTM model.
    Args:
        Y_train (list): list of labels.
        emb_matrix (dict): dictionary with words and their corresponding embeddings.
        learning_rate (float): learning rate for LSTM model.
        dropout (float): dropout rate for LSTM layer.
        optimizer (str): set optimizer.
        additional_dense (bool): whether to use Dense layer after the Embedding layer.
        bidirectional (bool): whether to use Bidirectional LSTM.

    Returns:
        model: LSTM model.
    """

    loss_function = "binary_crossentropy"

    if optimizer == "Adam":
        optim = Adam(learning_rate=learning_rate)
    else:
        optim = SGD(learning_rate=learning_rate)

    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = len(set(Y_train))
    
    # Now build the model
    model = Sequential()
    model.add(
        Embedding(
            num_tokens,
            embedding_dim,
            embeddings_initializer=Constant(emb_matrix),
            trainable=False,
        )
    )

    if additional_dense:
        model.add(Dense(units=hidden_size))

    if bidirectional:
        model.add(
            Bidirectional(LSTM(units=hidden_size, return_sequences=True, dropout=dropout))
        )
        model.add(Bidirectional(LSTM(units=hidden_size)))
    else:
        model.add(LSTM(units=hidden_size, return_sequences=True, dropout=dropout))
        model.add(LSTM(units=hidden_size))

    # Ultimately, end with dense layer with softmax
    model.add(Dense(units=1, activation="sigmoid"))
    # Compile model using our settings, check for accuracy
    model.compile(loss=loss_function, optimizer=optim, metrics=[f1])
    return model


def train_model(model, epochs, X_train, Y_train, X_dev, Y_dev, batch_size):
    """
    Function to train the model.
    Args:
        model: LSTM model to train.
        epochs (int): number of epochs.
        X_train (list): train documents.
        Y_train (list): train_labels.
        X_dev (list): development documents.
        Y_dev (list): development labels.
        batch_size (int): batch size.

    Returns:
        model: trained model.
    """

    verbose = 1
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It's also possible to monitor the training loss with monitor="loss"
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
    # Finally fit the model to our data
    model.fit(
        X_train,
        Y_train,
        verbose=verbose,
        epochs=epochs,
        callbacks=[callback],
        batch_size=batch_size,
        validation_data=(X_dev, Y_dev),
    )
    # Print final accuracy for the model (clearer overview)
    test_set_predict(model, X_dev, Y_dev, "dev")
    return model


def test_set_predict(model, X_test, Y_test, ident):
    """
    Make predictions and measure accuracy on our own test set (that we split off train).
    Args:
        model: trained model.
        X_test (list): test documents.
        Y_test (list): test labels.
        ident (str): name of the test set.

    Returns:
        Y_pred (list): list of predictions.
    """

    # Get predictions using the trained model
    Y_pred = model.predict(X_test)
    # Finally, convert to numerical labels to get scores with sklearn
    #Y_pred = np.argmax(Y_pred, axis=1)
    Y_pred=np.transpose(Y_pred)[0]  # transformation to get (n,)
    # Applying transformation to get binary values predictions with 0.5 as thresold
    Y_pred = list(map(lambda x: 0 if x<0.5 else 1, Y_pred))
    Y_pred = np.array(Y_pred, dtype=np.int64)
    # If you have gold data, you can calculate accuracy
    Y_test = np.array(Y_test, dtype=np.int64)
    print(Y_test[0])
    print(Y_pred[0])

    f1 = f1_score(Y_test.flatten(), Y_pred, average="macro")
    print("F1 macro on own {1} set: {0}".format(round(f1, 3), ident))
    return Y_test, Y_pred


def f1(y_true, y_pred):
    """
    F1 metric computation (because f1_score from sklearn.metrics doesn't work
    when training model).

    Args:
        y_true (list): list of true labels.
        y_pred (list): list of predicted labels.

    Returns:
        f1_score (float): F1 metric.
    """

    def recall(y_true, y_pred):
        """
        Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.

        Args:
            y_true (list): list of true labels.
            y_pred (list): list of predicted labels.

        Returns:
            recall (float): recall metric.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())

        return recall


    def precision(y_true, y_pred):
        """
        Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.

        Args:
            y_true (list): list of true labels.
            y_pred (list): list of predicted labels.

        Returns:
            precision (float): precision metric.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())

        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    f1_score = 2*((precision*recall)/(precision+recall+K.epsilon()))

    return f1_score


if __name__ == "__main__":
    """
    Main function to train and test neural network given cmd line arguments.
    """

    args = create_arg_parser()

    # Read in the data and embeddings
    train_documents, Y_train = read_corpus(args.train_file)
    dev_documents, Y_dev = read_corpus(args.dev_file)

    if args.prep:
        train_documents = preprocessing(train_documents)
        dev_documents = preprocessing(dev_documents)

    embeddings = read_embeddings(args.embeddings)

    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    dropout = args.dropout
    optimizer = args.optimizer
    additional_dense = args.add_dense
    bidirectional = args.bilstm
    hidden_size = args.hidden_size
    maxlen = args.maxlen

    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=maxlen)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(train_documents + dev_documents)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.transform(Y_dev)
    # Create model
    model = create_model(
        Y_train,
        emb_matrix,
        learning_rate,
        dropout,
        optimizer,
        additional_dense,
        bidirectional,
        hidden_size
    )

    # Transform input to vectorized input
    print(train_documents[0])
    X_train_vect = vectorizer(np.array([[s] for s in train_documents])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in dev_documents])).numpy()

    # Train the model
    model = train_model(
        model, epochs, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin, batch_size
    )

    # Do predictions on specified test set
    if args.test_file:
        # Read in test set and vectorize
        test_documents, Y_test = read_corpus(args.test_file)

        if args.prep:
            test_documents = preprocessing(test_documents)

        Y_test_bin = encoder.transform(Y_test)
        X_test_vect = vectorizer(np.array([[s] for s in test_documents])).numpy()

        # Finally do the predictions
        Y_test, Y_pred = test_set_predict(model, X_test_vect, Y_test_bin, "test")

        # Saving output
        data = {'text': test_documents, 'label': encoder.inverse_transform(Y_pred)} 
        df = pd.DataFrame(data)
        
        df.to_csv('output/LSTM/lstm_preds.tsv', sep="\t", header=False) 
