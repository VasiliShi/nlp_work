# -*- coding: utf-8 -*-
"""
Created on 2018/7/4

@author: Vasili
"""

from collections import Counter, deque
import re
from tqdm import tqdm
import tensorflow as tf
import numpy as np


class Word2Vec:
    def __init__(self, train_data, window=8, min_count=5, model="cbow", n_sampled=5, batch_size=128,
                 embedding_size=128):
        """

        :param train_data: 传入的数据list<line>
        :param window: 滑动窗口的大小
        :param min_count: 最小的词频
        :param model: cbow or skip-gram
        :param n_sampled: 采样词语的个数
        :param batch_size: batch的大小
        """
        self.train_data = train_data
        self.window = window
        self.min_count = min_count
        self.model = model
        self.n_sampled = n_sampled
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.point = 0
        # assert self.batch_size % self.n_sampled == 0
        self.cut()
        self.build_vocab()
        self.build_graph()
        self.train()

    def cut(self):
        import jieba
        self.data_words = []
        self.data_lines = []
        for line in tqdm(self.train_data):
            one_line = list(jieba.cut(line, cut_all=False))
            self.data_words.extend(one_line)
            self.data_lines.append(one_line)

    def clean(self):
        punct = ''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！）
        ，．＊：；Ｏ？｜｝︴︶︸︺︼︾﹀﹂﹄ ﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《
        「『【〔〖（［｛￡￥〝︵︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…０１２３４５６７８９'''
        stop_words = ""  # TODO 过滤停用词
        punct = set(punct)
        is_number = re.compile(r'\d+.*')
        filter_words = [w for w in self.data_words if (w not in punct) and (not is_number.search(w))]
        return filter_words

    def build_vocab(self):

        cleaned_words = self.clean()
        words_counter = Counter(cleaned_words)
        self.words = {word: count for word, count in words_counter.items() if count > self.min_count}
        self.words['UNK'] = -1
        self.word2id = {word: idx for idx, word in enumerate(self.words.keys())}
        self.id2word = dict(zip(self.word2id.values(), self.word2id.keys()))
        self.data = []
        unk = 0
        for w in cleaned_words:
            if w in self.word2id:
                index = self.word2id.get(w)  # 返回word的编号
            else:
                unk += 1
                index = self.word2id.get('UNK')
            self.data.append(index)
        self.words['UNK'] = unk

    def _generate_batch_sg(self):
        """
        使用采样的方法生成训练batch
        :return:
        """
        assert self.n_sampled <= 2 * self.window

        batch = np.zeros(shape=(self.batch_size,), dtype=np.int32)
        labels = np.zeros(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.window + 1  #
        buffer = deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.point])
            self.point = (self.point + 1) % len(self.data)  # 需要多次迭代数据集，循环访问

        for i in range(self.batch_size // self.n_sampled):  # 中心词语的数量，个数是自己设定的
            rand_x = np.random.permutation(span)
            j, k = 0, 0
            for j in range(self.n_sampled):  # 采样的个数
                if rand_x[k] == self.window:  # 这个就是中心词
                    k += 1
                batch[i * self.n_sampled + j] = buffer[self.window]
                labels[i * self.n_sampled + j, 0] = buffer[rand_x[k]]
                k += 1
            rand_step = np.random.randint(1, 5)
            for _ in range(rand_step):
                buffer.append(self.data[self.point])
                self.point = (self.point + 1) % len(self.data)
        return batch, labels

    def _generate_batch_cbow(self):
        """
        生成训练batch
        :return:
        """
        num_context = self.n_sampled * self.batch_size
        batch = np.zeros(shape=(num_context), dtype=np.int32)
        labels = np.zeros(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.window + 1
        buffer = deque(maxlen=span)
        for _ in range(span):
            buffer.append(self.data[self.point])  # 这个地方大意写错了
            self.point = (self.point + 1) % len(self.data)

        for i in range(self.batch_size):  # n_sampled个单词组成的语境，batch_size个滑动窗口
            rand_x = np.random.permutation(span)
            if 2 * self.window == self.n_sampled:
                for j in range(self.window):
                    batch[i * self.n_sampled + j] = buffer[j]
                for j in range(self.window, 2 * self.window):
                    batch[i * self.n_sampled + j] = buffer[j + 1]
            else:
                j, k = 0, 0
                for j in range(self.n_sampled):
                    if rand_x[k] == self.window:
                        k += 1

                    batch[i * self.n_sampled + j] = buffer[rand_x[k]]  # 这个地方写错了，找了半天错误=。=
                    k += 1
            labels[i, 0] = buffer[self.window]  # 添加中心词到label

            rand_step = np.random.randint(1, 5)
            for _ in range(rand_step):
                buffer.append(self.data[self.point])
                self.point = (self.point + 1) % len(self.data)

        return batch, labels

    def build_graph(self):
        vocab_size = len(self.word2id)
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope("input_value"):

                if self.model == "skip-gram":  # 判断模型的种类
                    self.train_x = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
                    self.train_y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, 1])
                else:
                    self.train_x = tf.placeholder(dtype=tf.int32, shape=[self.batch_size * self.n_sampled])
                    self.train_y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, 1])

            self.embedding = tf.Variable(tf.random_uniform([vocab_size, self.embedding_size], -1.0, 1.0),
                                         name="embedding")
            print("self.embedding", self.embedding.shape.as_list())
            softmax_weight = tf.Variable(
                tf.truncated_normal(
                    [vocab_size, self.embedding_size], stddev=1.0 / np.sqrt(self.embedding_size)), name="weight")
            softmax_bias = tf.Variable(tf.zeros([vocab_size]), name="bias")

            embed = tf.nn.embedding_lookup(self.embedding, self.train_x, name="embed")
            if self.model == "cbow":
                embed = tf.reshape(embed, [self.batch_size, self.n_sampled, -1])
                embed = tf.reduce_mean(input_tensor=embed, axis=1, keepdims=False)

            self.loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(
                    weights=softmax_weight,
                    biases=softmax_bias,
                    labels=self.train_y,
                    inputs=embed,
                    num_sampled=64,
                    num_classes=vocab_size
                )
            )
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
            norm = tf.sqrt(tf.reduce_mean(tf.square(self.embedding), 1, keepdims=True))
            self.normalized_embedding = self.embedding / norm

    def most_similar(self, word, topn=10):
        """
        余弦相似度 cos = ab/|a||b|
        :param word: 输入的词语
        :param topn: 取前N个
        :return:
        """
        if word not in self.word2id:
            raise Exception(f"{word} not in the VOCAB")
        idx = self.word2id[word]
        word_vec = self.normalized_embedding[idx]
        simiarity = np.matmul(word_vec, np.transpose(self.normalized_embedding))
        sim_arg = simiarity.argsort()[::-1]  # 逆序从大到小
        return [(self.id2word[i], simiarity[i]) for i in sim_arg[1:topn + 1]]  # 排除掉第一个

    def train(self):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            for step in tqdm(range(500)):
                if self.model == "skip-gram":
                    batch_data, batch_label = self._generate_batch_sg()
                else:
                    batch_data, batch_label = self._generate_batch_cbow()
                feed_dict = {self.train_x: batch_data, self.train_y: batch_label}
                _, l = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                # print(f"training epoch {step}")
            self.embedding, self.normalized_embedding = sess.run([self.embedding, self.normalized_embedding])


if __name__ == '__main__':
    file_name = "data/ice_and_fire1.txt"
    text_lines = []
    with open(file_name) as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            text_lines.append(line)
        print(f"总共读入了{len(text_lines)}行数据")
    wv = Word2Vec(text_lines, min_count=1)
