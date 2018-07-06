# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 18:34:57 2017

@author: VasiliShi
"""

import tensorflow as tf
from seq2seq_char import source_vocab_to_idx, source_idx_to_vocab

batch_size = 128


def source_to_seq(text):
    seq_len = 7
    result = [source_vocab_to_idx.get(w, source_vocab_to_idx.get('UNK')) \
              for w in text] + [source_vocab_to_idx['<PAD>']] * (seq_len - len(text))
    return result


checkpoint = "model/seq_model.ckpt"
raw_input = "edcba"
text = source_to_seq(raw_input)

graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    model = tf.train.import_meta_graph(checkpoint + '.meta')
    model.restore(sess, checkpoint)
    input_data = graph.get_tensor_by_name('inputs:0')
    logists = graph.get_tensor_by_name('predictions:0')
    source_sequence_length = graph.get_tensor_by_name("source_sequence_length:0")
    target_sequence_length = graph.get_tensor_by_name("target_sequence_length:0")
    result = sess.run(logists, {input_data: [text] * batch_size,
                                target_sequence_length: [len(raw_input)] * batch_size,
                                source_sequence_length: [len(raw_input)] * batch_size
                                })[0]  # 得到的是一个 batch_size * seq_len的矩阵

pad = source_vocab_to_idx['<PAD>']
print('原始输入：', raw_input)

print("Source:")
print('word 编号 {}'.format([i for i in text]))
print('组成字母{}'.format([source_idx_to_vocab[i] for i in text]))

print("Target")
print('word 编号 {}'.format([i for i in result]))
print("组成字母{}".format([source_idx_to_vocab[i] for i in result]))
