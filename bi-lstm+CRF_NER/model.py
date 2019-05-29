# -*- coding:utf-8 -*- 
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood,viterbi_decode
import  time
from util import get_logger
from data import batch_yield,pad_sequences
import  sys
import  os
from eval import  conlleval

class BiLSTM_CRF(object):

    def __init__(self, args, embedding, tag2label, vocab, paths, config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch 
        self.hidden_dim = args.hidden_dim
        self.CRF = args.CRF
        self.clip_grad = args.clip
        self.embedding = embedding
        self.dropout_keep_prob = args.dropout
        self.update_embedding = args.update_embedding
        self.shuffle = args.shuffle
        self.tag2label = tag2label
        self.num_tags = len(tag2label)
        self.lr = args.lr
        self.vocab = vocab
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.config = config

    def build_graph(self):
        self.add_placeholders()
        self.lookup_layer()
        self.bi_lstm_layer()
        self.model_loss()
        self.trainstep()

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name='word_ids')
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],name='sequence_lengths' )
        
        self.dropout_pl = tf.placeholder(tf.float32, shape=[], name='dropout')
        self.lr_pl = tf.placeholder(tf.float32, shape=[], name='lr')
        
    def lookup_layer(self):
        with tf.variable_scope('words'):
            weight = tf.Variable(self.embedding, 
                                         dtype=tf.float32,
                                         trainable= self.update_embedding,
                                         name = 'embedding')
            word_embeddings = tf.nn.embedding_lookup(params=weight,
                                                     ids=self.word_ids,
                                                     name='word_embedding')
        self.word_embedding = tf.nn.dropout(word_embeddings, self.dropout_pl)
        
    def bi_lstm_layer(self):
        with tf.variable_scope('bi-lstm'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(num_units= self.hidden_dim)
            cell_bw = tf.nn.rnn_cell.LSTMCell(num_units= self.hidden_dim)
            (fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw = cell_fw,
                    cell_bw = cell_bw,
                    inputs = self.word_embedding,
                    sequence_length = self.sequence_lengths,
                    dtype=tf.float32)
            output = tf.concat([fw_output, bw_output], axis=-1)
            output = tf.nn.dropout(output, self.dropout_pl)
            
        with tf.variable_scope('projection'):
            W = tf.get_variable(name='W', 
                                shape=[2* self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable(name='b',shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),
                                dtype=tf.float32)

            s = tf.shape(output)
            output = tf.reshape(output, [-1, 2*self.hidden_dim])
            pred = tf.matmul(output, W) + b
            self.logit = tf.reshape(pred, [-1, s[1], self.num_tags])

    def model_loss(self):

        if self.CRF:
            log_likelihood ,self.transition_params = crf_log_likelihood(inputs=self.logit,
                                                                        tag_indices=self.labels,
                                                                        sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logit,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)


        tf.summary.scalar('loss', self.loss)

    def softmax_pred(self):
        if not self.CRF:
            self.label_softmax = tf.argmax(self.logit, axis=-1)
            self.label_softmax = tf.cast(self.label_softmax, tf.int32)

    def trainstep(self):
        with tf.variable_scope('train_step'):
            self.glob_step = tf.Variable(0, name='glob_step',trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate= self.lr)
            # 步骤1
            grads_and_vars = optimizer.compute_gradients(self.loss)

            # 步骤2
            grads_and_vars_clips = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g,v in grads_and_vars]
            # 步骤3
            self.train_op = optimizer.apply_gradients(grads_and_vars_clips, global_step=self.glob_step)
    def summery(self,sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config = self.config) as sess:
            sess.run(tf.global_variables_initializer())
            self.summery(sess)
            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)

    def run_one_epoch(self,sess, train, dev, tag2label, epoch, saver):
        num_batches = (len(train) + self.batch_size - 1)  // self.batch_size # 目的是为了上取整
        start_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
        batches = batch_yield(train, self.batch_size,self.vocab, self.tag2label, shuffle=self.shuffle)

        for step,(seqs,labels) in enumerate(batches):
            sys.stdout.write(f'precessing {step + 1} batch / {num_batches} batches\r')
            # 当前总共走了多少步
            step_num = epoch * num_batches + step + 1
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)

            _, loss_train, summery,step_nu = sess.run([self.train_op, self.loss,self.merged, self.glob_step],
                                                       feed_dict = feed_dict)
            if (step + 1) == 1 or (step + 1) % 300 == 0 or (step + 1) == num_batches:
                self.logger.info(f'{start_time}: epoch {epoch+1}, '
                                 f'step {step+1}, loss:{loss_train:.4}, global step:{step_num}')
            self.file_writer.add_summary(summary=summery,global_step=step_num) # 总步

            if step + 1 == num_batches:# 最后一个batch的时候
                saver.save(sess, self.model_path, global_step=step_num)

        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        # self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)


    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):

        word_ids,seq_len_list = pad_sequences(seqs)
        feed_dict = {self.word_ids:word_ids,
                     self.sequence_lengths:seq_len_list}
        if labels is not None:
            labels_tmp,_ = pad_sequences(labels)
            feed_dict[self.labels] = labels_tmp

        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list


    def dev_one_epoch(self,sess, dev):
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        # 预测阶段 dropout的值
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)

        if self.CRF:
            logits, transition_params = sess.run([self.logit, self.transition_params],
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
                label_list.append(viterbi_seq)
            return label_list, seq_len_list
        else:
            label_list = sess.run(self.label_softmax, feed_dict=feed_dict)
            return label_list, seq_len_list

    def demo_one(self, sess, sent):

        label_list = []
        for seqs,labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            predict_list,_ = self.predict_one_batch(sess, seqs)

            print(f"***********{predict_list}************")
            label_list.extend(predict_list)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label !=0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def test(self,test):
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========testing========')
            saver.restore(sess, self.model_path)
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            self.evaluate(label_list, seq_len_list, test)

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """

        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if  len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path):
            self.logger.info(_)









            
            
            
            
            
    