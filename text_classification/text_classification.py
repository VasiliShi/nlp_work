#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 21:19:20 2019

@author: vasili
"""

import pandas as pd
import time
import numpy as np
emotion_df = pd.read_csv('text_emotion.csv')
print(f'total length of dataset is :{len(emotion_df)}')
print(emotion_df['sentiment'].value_counts())

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

VOCAB_SIZE = 50000

tfidf_vec = TfidfVectorizer(max_features=VOCAB_SIZE,min_df=2)
label_encoder = LabelEncoder()

##fit and transform
X = tfidf_vec.fit_transform(emotion_df['content'])
y = label_encoder.fit_transform(emotion_df['sentiment'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## naive bayes
bayes_clf = MultinomialNB()
bayes_clf.fit(X_train, y_train)
predictions = bayes_clf.predict(X_test)
bayes_precision = precision_score(predictions, y_test, average='micro')

'''
classifiers = {'sgd': SGDClassifier(loss='hinge'),
              # 'svm': SVC(), # too slowly
               'random_forest': RandomForestClassifier()}

for clf_name, clf in classifiers.items():
    start = time.time()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    end = time.time()
    print(clf_name, precision_score(predictions, y_test, average='micro'),f"cost time :{end - start}")
  
'''
d = np.eye(len(tfidf_vec.vocabulary_))
word_predict = bayes_clf.predict_proba(d) # 预测每一个词语
id2word = {idx : word for word,idx in tfidf_vec.vocabulary_.items()}

# 查看每一个word对于每一个类别的概率，
from collections import Counter ,defaultdict
by_cls = defaultdict(Counter)
for word_idx ,pred in enumerate(word_predict):# word_idx 
    for class_idx, score in enumerate(pred):
        cls = label_encoder.classes_[class_idx]
        by_cls[cls][id2word[word_idx]] = score
        
    
for k in by_cls:
    words = [x[0] for x in by_cls[k].most_common(5)] # most_common按照value进行排序
    print(k, ':', ' '.join(words))



## try deep model,前面使用了简单的线性模，这里我们并没有太多的数据，这样会限制模型的效果，因为使用深度模型会导致over fitting
from itertools import chain
from keras.preprocessing.sequence import pad_sequences

#content = emotion_df['content'].apply(lambda x:x.split()) # word-level
chars = list(sorted(set(chain(*emotion_df['content'])))) # char-level
char2id = {char:idx for idx,char in enumerate(chars)}
max_seq_len = max(len(x) for x in emotion_df['content'])
char_vectors = []
for txt in emotion_df['content']:
    vec = np.zeros((max_seq_len,len(char2id)))
    vec[np.arange(len(txt)), [char2id[c] for c in txt]] = 1
    char_vectors.append(vec)
char_vectors = np.array(char_vectors)
char_vectors = pad_sequences(char_vectors)

labels = label_encoder.fit_transform(emotion_df['sentiment'])

def split(data,ratio = 0.9):
    train_cnt = int(ratio * len(data))
    return data[:train_cnt],data[train_cnt:]

X_train, X_test, y_train, y_test = train_test_split(
        char_vectors, labels, test_size=0.3, random_state=42)
