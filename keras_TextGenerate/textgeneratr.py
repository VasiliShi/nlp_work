#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 15:32:07 2019

@author: vasili
TextGenerate or CharRNN
"""

import numpy as np
import keras
from keras import layers
import random
import sys

path = keras.utils.get_file('nicai.txt',origin = 'https://s3.amazonaws.com/text-datasets/nietzsche.txt')

text = open(path).read().lower()
print(f"Corpus length is {len(text)}")
def reweight_distribution(original_distribution, temperature = 0.5):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)

maxlen = 60
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i + maxlen]) #maxlen长度的sent
    next_chars.append(text[i + maxlen])
print("Number of sequences:",len(sentences))

chars = sorted(list(set(text)))  
print("Unique characters:",len(chars))
char_id = dict((char,i)for i,char in enumerate(chars))

print("ont-hot Vectorization...")
x = np.zeros((len(sentences),maxlen,len(chars)),dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t,char in enumerate(sentence):
        x[i,t,char_id[char]] = 1 # 该位置的值为True
    y[i, char_id[next_chars[i]]] = 1 # label

# optional
def char_rnn_model(num_chars, num_layers, num_nodes = 512, dropout=0.1):
    inputs = Input(shape=(None,num_chars), name='input')
    prev = inputs
    for i in range(num_layers):
        lstm = LSTM(num_nodes, return_sequences=True, name=f'lstm_layer_{(i+1)}')(prev)
        if dropout:
            prev = Dropout(dropout)(lstm)
        else:
            prev = lstm
    dense = TimeDistributed(Dense(num_chars, name='dense', activation='softmax'))(prev)
    model = Model(inputs = [inputs], outputs = [dense])
    optimizer = RMSprop(lr = 0.01)
    model.compile(optimizer=optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

model = keras.models.Sequential()

model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation = 'softmax'))
optimizer = keras.optimizers.RMSprop(lr = 0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature = 1.0):
    """
    根据每个char的概率，找出下一个输出的char
    """
    preds = np.array(preds, dtype=np.float64)
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1) # 
    return np.argmax(probas)
for epoch in range(1,20):
    print(f"Epoch:{epoch}")
    model.fit(x,y,batch_size = 128, epochs=1) # 注意这个地方epoch 为1，只在数据机上面拟合一次
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index:start_index+maxlen]
    print("****** Generating with seed: '" + generated_text + "'")
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print(f"***Temperature: {temperature}***")
        sys.stdout.write(generated_text)
        for i in range(400): #生成400个字符
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0,t,char_id[char]] = 1
                
            preds = model.predict(sampled, verbose=0)[0] # inference 阶段,去除第一行
            #next_index = np.argmax(preds)
            #next_char = chars[next_index]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            generated_text += next_char[1:] #舍弃第一个字符作为下一轮inference，
            sys.stdout.write(next_char)


