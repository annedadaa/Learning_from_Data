### Predicted values

Folder that contains texts and predictions based on the best models of different types: SVM, LSTM, pretrained roBERTa. 

All the texts are preprocessed using [preprocessing function](https://github.com/annedadaa/Offensive_Language_Identification/blob/d5d9b8599bced982179c9acd55cf9c99428432f5/train_predict/LanguageModels.py#L83).

Because SVM and LSTM served as baselines, LMs results interested us more and showed (as expected) the best results. Thus, we provide 4 output files for 4 different models tested (filenames consist of model-name_preds.tsv). We also provide outputs of the best roBERTa model trained on the additional Toxic Tweets dataset (tuned_on_toxic_data_preds.tsv).
