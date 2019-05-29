# -*- coding: utf-8 -*-
"""
Created on 2019-04-21

@author: Vasili
"""

import  random
import  pickle
import  numpy as np

tag2label = {
    "O": 0,
    "B-PER": 1, "I-PER": 2,
    "B-LOC": 3, "I-LOC": 4,
    "B-ORG": 5, "I-ORG": 6
}
def sentence2id(sent, word2id):
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    if shuffle:
        random.shuffle(data)
    seqs, labels = [], []
    for (sent, tag) in data:
        sent = sentence2id(sent, vocab)
        label = [tag2label[t] for t in tag]
        seqs.append(sent)
        labels.append(label)
        if len(seqs) == batch_size:
            yield seqs, labels
            seqs,labels = [], []

    if len(seqs) != batch_size:
        yield seqs, labels

def pad_sequences(sequences, mask=0):
    max_len = max(map(len, sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_tmp = seq[:max_len] + [mask] * max((max_len - len(seq)), 0)
        seq_list.append(seq_tmp)
        seq_len_list.append(min(len(seq), max_len)) #不会出现这种情况

    return seq_list, seq_len_list


def read_corpus(corpus_path):
    data = []
    with open(corpus_path,encoding='utf-8') as f:
        word,tag = [], []
        for line in f:
            if line != '\n':
                w,t = line.strip().split()
                word.append(w)
                tag.append(t)
            else:
                data.append((word,tag))
                word ,tag= [],[]
    return data

def read_dictionary(vocab_path):
    with open(vocab_path, 'rb') as f:
        word2id = pickle.load(f)
    print(f'loading vocab success,vocab size {len(word2id)}')
    return word2id

def random_embedding(vocab, embedding_dim):
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat) # 如果没有这个会怎样
    return  embedding_mat




if __name__ == '__main__':
    pass