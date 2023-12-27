#!/usr/bin/env python
# coding: utf-8

# # Team Members  
# ## 1. Mahmoud Salah Ahmed `20180254`
# ## 2. Alaa Eldin Ebrahim `20200330`
# ## 3. Hana Hany Ayman `20201213`
# ## 4. Donia Ahmed Abo Zeid `20201060`

# # Required imports



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import string
import warnings




warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="scikeras")
warnings.filterwarnings("ignore", message=".*'token_pattern'.*")


# # Data Preprocessing



nlp = spacy.load('en_core_web_sm')
stopwords = list(STOP_WORDS)
stopwords.remove('not')




data = pd.read_csv('sentimentdataset (Project 1).csv')
print(data.head(10))

data = data.drop(columns=['ID', 'Source'])

print(data['Target'].value_counts())


def text_data_cleaning(sentence):
    doc = nlp(sentence)

    tokens = []  # list of tokens
    for token in doc:
        if token.lemma_ != "-PRON-":
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)

    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords and token not in string.punctuation:
            cleaned_tokens.append(token)
    return cleaned_tokens


X = data['Message']
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# # Linear SVC

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=text_data_cleaning)),
    ('clf', LinearSVC()),
])


param_grid = {
    'tfidf__max_df': [0.5, 0.75, 1.0],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'clf__C': [0.1, 1, 10],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

y_pred = grid_search.predict(X_test)


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

import joblib
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'linear_svm_best_model.joblib')


loaded_model = joblib.load('linear_svm_best_model.joblib')

new_data = ["it is very good", 'it is bad', 'awesome', 'I am not comfortable with that']
predictions = loaded_model.predict(new_data)

print(predictions)


# # ANN

from sklearn.neural_network import MLPClassifier


data['processed_text'] = data['Message'].apply(text_data_cleaning)

data['processed_text'] = [' '.join(sentence) for sentence in data['processed_text']]




print(data)

data = data.drop(['Message'], axis=1)


X = data['processed_text']
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


parameters = {
    'hidden_layer_sizes': [(50,), (100,), (128,), (256,), (512,)],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
}

tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

ann_model = MLPClassifier(max_iter=500)


grid_search = GridSearchCV(estimator=ann_model, param_grid=parameters, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)

best_params = grid_search.best_params_

print(best_params)


best_ann_model = MLPClassifier(max_iter=500, **best_params)
best_ann_model.fit(X_train_tfidf, y_train)

y_pred = best_ann_model.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f'Accuray: {accuracy_score(y_pred, y_test)}')


import joblib
joblib.dump(best_ann_model, 'best_ann_model.joblib')


loaded_model = joblib.load('best_ann_model.joblib')
new_data = ["This is a positive sentence.", "This is a negative sentence."]
processed_new_data = [text_data_cleaning(sentence) for sentence in new_data]
processed_new_data = [' '.join(sentence) for sentence in processed_new_data]
new_data_tfidf = tfidf_vectorizer.transform(processed_new_data)
predictions = loaded_model.predict(new_data_tfidf)
print("Predictions on new data:")
print(predictions)



