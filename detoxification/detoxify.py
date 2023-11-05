#!/usr/bin/env python

"""Detoxification of offensive texts"""

# Import all necessary libraries.
import pandas as pd
import argparse
from transformers import pipeline, BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

def create_arg_parser():
    parser = argparse.ArgumentParser()

    # Filepath arguments
    parser.add_argument("-tr", "--corpus_path", default="output/LMs/twitter-roberta-base-sentiment-latest_preds.tsv", type=str,
                        help="Input file to learn from (default train.tsv)",)

    # Whether to paraphrase toxic wordings. Otherwise prepared dataset will be used 
    parser.add_argument("-d", "--detox", action="store_true", help="Whether to detoxify texts",)


    args = parser.parse_args()
    return args


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

    offensive_documents = corpus[corpus['label'] == "OFF"]
    
    documents = offensive_documents['text'].to_list()
    labels = offensive_documents['label'].to_list()

    return documents, labels


def get_embeddings(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs.to("cuda"))
        bert_embedding = outputs.last_hidden_state.cpu().mean(dim=1)

    return bert_embedding


if __name__ == "__main__":
    args = create_arg_parser()

    idx_to_label = {'LABEL_0': 'NOT', 'LABEL_1': 'OFF'}

    offensive_documents, offensive_labels = read_corpus(args.corpus_path)

    # Detoxify texts
    if args.detox:
        pipe = pipeline("text2text-generation", model="s-nlp/bart-base-detox")
        detoxified_documents = []
        for output in pipe(offensive_documents):
            detoxified_documents.append(output['generated_text'])

        # Get BERT embeddings for two sentences 
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        model = BertModel.from_pretrained("bert-base-cased").to("cuda")

        cos_sims = []

        for n, text in enumerate(offensive_documents):
            emb_sent1 = get_embeddings(text, model, tokenizer)
            emb_sent2 = get_embeddings(detoxified_documents[n], model, tokenizer)
            cos_sims.append(cosine_similarity(emb_sent1, emb_sent2)[0][0])

        data = {'original': offensive_documents, 'detoxified': detoxified_documents, 'similarity': cos_sims,
                'orig_label': offensive_labels} 
        df = pd.DataFrame(data)
        df.to_csv('detoxification/detoxified.csv')

    else:
        detoxified = pd.read_csv('detoxification/detoxified.csv')

        # Threshold to ensure that texts are still similar in mearning
        detoxified_rows = detoxified[(detoxified['similarity'] >= 0.50) & (detoxified['similarity'] <= 0.90)]
        print("Number of detoxified rows:", len(detoxified_rows))

        # Classify whether new detoxified text is still offensive 
        offensive_detection = pipeline("text-classification", model="train_clf/results/twitter-roberta-base-sentiment-latest/checkpoint-best-79.5")
        preds = offensive_detection(detoxified_rows['detoxified'].to_list())
        
        preds = [idx_to_label[output['label']] for output in preds]

        # Save results
        data = {'original': detoxified_rows['original'].to_list(), 'detoxified': detoxified_rows['detoxified'].to_list(),
                'orig_label': detoxified_rows['orig_label'].to_list(), 'new_label': preds} 
        df = pd.DataFrame(data)
        df.to_csv('detoxification/detoxified_final.csv')
