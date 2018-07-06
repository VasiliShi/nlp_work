# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:45:17 2017

@author: VasiliShi
"""

import numpy as np

import tensorflow as tf

# 超参数
# Number of Epochs# Number
epochs = 60
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.001

source_data = []
target_data = []
with open("data/letters_source.txt", encoding='utf-8') as f:
    for line in f:
        source_data.append(line.strip())
with open("data/letters_target.txt", encoding='utf-8') as f:
    for line in f:
        target_data.append(line.strip())


def build_vocab(data):
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    set_chars = list(set([char for line in data for char in line]))
    idx_to_vocab = {idx: char for idx, char in enumerate(special_words + set_chars)}
    vocab_to_idx = {char: idx for idx, char in enumerate(special_words + set_chars)}
    return idx_to_vocab, vocab_to_idx


source_idx_to_vocab, source_vocab_to_idx = build_vocab(source_data)
target_idx_to_vocab, target_vocab_to_idx = build_vocab(target_data)
source_idx = [
    [source_vocab_to_idx.get(char, source_vocab_to_idx.get('<UNK>')) for char in line]
    for line in source_data]
target_idx = [
    [target_vocab_to_idx.get(char, target_vocab_to_idx.get('<UNK>')) for char in line]
    for line in target_data]


def build_placeholders():
    """
    """
    inputs = tf.placeholder(tf.int32, [None, None], name="inputs")
    targets = tf.placeholder(tf.int32, [None, None], name="targets")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    target_sequence_length = tf.placeholder(tf.int32, (None,), name="target_sequence_length")

    max_target_sequence_length = tf.reduce_max(target_sequence_length, name="max_target_len")
    source_sequence_length = tf.placeholder(tf.int32, (None,), name="source_sequence_length")
    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length


def encoder_layer(input_data, rnn_size, num_layers, source_sequence_length,
                  source_vocab_size, encoding_embedding_size):
    encoder_embed_input = tf.contrib.layers.embed_sequence(
        input_data,
        source_vocab_size,
        encoding_embedding_size)

    def lstm_cell(rnn_size):
        cell = tf.contrib.rnn.LSTMCell(
            rnn_size,
            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2)
        )
        return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell(rnn_size) for i in range(num_layers)]
    )
    encoder_output, encoder_state = tf.nn.dynamic_rnn(
        cell,
        encoder_embed_input,
        sequence_length=source_sequence_length,
        dtype=tf.float32
    )
    return encoder_output, encoder_state


# 对target的数据进行处理
def process_decoder_input(data, vocab_to_idx, batch_size):
    ending = tf.strided_slice(data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], vocab_to_idx['<GO>']), ending], 1)
    return decoder_input


def decoder_layer(target_vocab_to_idx, decoding_embedding_size,
                  num_layers, rnn_size, target_sequence_length,
                  max_target_sequence_length, encoder_state, decoder_input):
    # 1.embedding
    target_vocab_size = len(target_vocab_to_idx)
    decoder_embedding = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embedding, decoder_input)

    # 2.decoder rnn
    def lstm_cell(rnn_size):
        cell = tf.contrib.rnn.LSTMCell(
            rnn_size,
            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2)
        )
        return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell(rnn_size) for i in range(num_layers)])
    # 3.output layer
    output_layer = tf.layers.Dense(target_vocab_size, use_bias=False)
    # 4.Training decoder
    with tf.variable_scope("decode"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(
            inputs=decoder_embed_input,
            sequence_length=target_sequence_length)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell,
            helper=training_helper,
            initial_state=encoder_state,
            output_layer=output_layer)

        training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            training_decoder,
            impute_finished=True,
            maximum_iterations=max_target_sequence_length)
    # 5.predicting decoder
    with tf.variable_scope("decode", reuse=True):
        start_token = tf.tile(tf.constant([target_vocab_to_idx['<GO>']], dtype=tf.int32),
                              [batch_size],
                              name="start_tokens")
        predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            decoder_embedding,
            start_token,
            end_token=target_vocab_to_idx['<EOS>'])

        predicting_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=cell,
            helper=predicting_helper,
            initial_state=encoder_state,
            output_layer=output_layer
        )
        predicting_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            predicting_decoder,
            impute_finished=True,
            maximum_iterations=max_target_sequence_length)
    return training_decoder_output, predicting_decoder_output


# 构建完成encoder和decoder之后，开始构建seq2seq模型

def model(input_data, targets, lr, target_sequence_length,
          max_target_sequence_length, source_sequence_length,
          source_vocab_size, target_vocab_size, encoder_embedding_size,
          decoder_embedding_size, rnn_size, num_layers):
    # encoder
    _, encoder_state = encoder_layer(input_data,
                                     rnn_size,
                                     num_layers,
                                     source_sequence_length,
                                     source_vocab_size,
                                     encoder_embedding_size)
    # decoder的输入
    decoder_input = process_decoder_input(targets, target_vocab_to_idx, batch_size)
    training_decoder_output, predicting_decoder_output = decoder_layer(target_vocab_to_idx,
                                                                       decoder_embedding_size,
                                                                       num_layers,
                                                                       rnn_size,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       encoder_state,
                                                                       decoder_input)
    return training_decoder_output, predicting_decoder_output


# pad sentences
def pad_sentence_batch(sentence_batch, pad_int):
    """
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
    """
    max_sentence_len = max([len(sen) for sen in sentence_batch])
    return [sen + [pad_int] * (max_sentence_len - len(sen)) for sen in sentence_batch]


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    for batch in range(0, len(sources) // batch_size):
        start = batch * batch_size
        source_batch = sources[start:start + batch_size]
        target_batch = targets[start:start + batch_size]
        pad_source_batch = np.array(pad_sentence_batch(source_batch, source_pad_int))
        pad_target_batch = np.array(pad_sentence_batch(target_batch, target_pad_int))
        targets_length = []
        sources_length = []
        for target in target_batch:
            targets_length.append(len(target))
        for source in source_batch:
            sources_length.append(len(source))
        yield pad_target_batch, pad_source_batch, targets_length, sources_length


# Train 训练
train_source = source_idx[batch_size:]
train_target = target_idx[batch_size:]

valid_source = source_idx[:batch_size]
valid_target = target_idx[:batch_size]

# 注意这里面next的用法
"""
 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
def f(a):
    for i in a:
        yield  i
z = f(a)
next(z) => 0
next(z) => 1
"""
(valid_targets_batch,
 valid_sources_batch,
 valid_targets_length,
 valid_sources_length) = next(get_batches(valid_target, valid_source, batch_size,
                                          source_vocab_to_idx['<PAD>'],
                                          target_vocab_to_idx['<PAD>']))


def main():
    train_graph = tf.Graph()
    with train_graph.as_default():
        input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = build_placeholders()

        training_decoder_output, predicting_decoder_output = model(input_data,
                                                                   targets,
                                                                   lr,
                                                                   target_sequence_length,
                                                                   max_target_sequence_length,
                                                                   source_sequence_length,
                                                                   len(source_vocab_to_idx),
                                                                   len(target_vocab_to_idx),
                                                                   encoding_embedding_size,
                                                                   decoding_embedding_size,
                                                                   rnn_size,
                                                                   num_layers)
        # Return a tensor with the same shape and contents as input.                                                          )
        training_logits = tf.identity(training_decoder_output.rnn_output, name='logits')
        predicting_logits = tf.identity(predicting_decoder_output.sample_id, name="predictions")
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)
        optimizer = tf.train.AdamOptimizer(lr)
        # Gradient clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    display_step = 50
    checkpoint = "model/seq_model.ckpt"
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(1, epochs):
            for batch_i, (target_batch, source_batch, target_length, source_length) in enumerate(
                    get_batches(train_target, train_source, batch_size,
                                source_vocab_to_idx['<PAD>'],
                                target_vocab_to_idx['<PAD>'])):
                _, loss = sess.run(
                    [train_op, cost],
                    {input_data: source_batch,
                     targets: target_batch,
                     lr: learning_rate,
                     target_sequence_length: target_length,
                     source_sequence_length: source_length}
                )
                if batch_i % display_step == 0:
                    validation_loss = sess.run(
                        [cost],
                        {input_data: valid_sources_batch,
                         targets: valid_targets_batch,
                         lr: learning_rate,
                         target_sequence_length: target_length,
                         source_sequence_length: source_length
                         }
                    )
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                          .format(epoch_i,
                                  epochs,
                                  batch_i,
                                  len(train_source) // batch_size,
                                  loss,
                                  validation_loss[0]))
        saver = tf.train.Saver()
        saver.save(sess, checkpoint)
        print("model has been Saved")


if __name__ == "__main__":
    main()
