# -*- coding: utf-8 -*-
"""
This is based on model in paper 'Convolutional Neural Networks for Sentence Classification'

single-channel CNN model for text classification

"""

from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf


class CNNTextModel(object):
    def __init__(self, vocab_size, max_seq_length, num_classes, batch_size, embedding_size,
                 num_filters, filter_widths, learning_rate=0.05, max_grad_norm=3.0, embedding_static=True, embedding_init=None):
        print('Initialize new model')
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.num_filters = num_filters
        self.filter_widths = filter_widths
        assert(len(num_filters)==len(filter_widths))

        self.lr = learning_rate
        self.max_grad_norm = max_grad_norm

        self.embedding_static = embedding_static
        self.embedding_init = embedding_init
        assert( (not embedding_static or embedding_init is not None) )


    def _create_placeholder(self):
        '''
        Feeds for inputs
        :return:
        '''
        print('Create placeholders')
        self.inputs = tf.placeholder(tf.int32, shape=[None, self.max_seq_length], name='inputs')
        self.targets = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='targets')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    def _inference(self):
        '''
        :return:
        '''
        print('Create inference')
        # embedding layer: learn embedding if embedding_static is False
        with tf.name_scope('embedding'):
            if self.embedding_init is None:
                self.embed_matrix = tf.Variable(tf.truncated_normal(
                    shape=[self.vocab_size, self.embedding_size]), name='embed_matrix')
            else:
                self.embed_matrix = tf.Variable(self.embedding_init, trainable=(not self.embedding_static), name='embed_matrix')

            self.embedding = tf.nn.embedding_lookup(self.embed_matrix, self.inputs, name='embedding')
            self.embedding = tf.expand_dims(self.embedding, -1)         # [batch_size x max_seq_length x embedding_size x 1]
        # convolution, max-pooling layer: single wide-cnn layer
        pools = []
        with tf.variable_scope('conv-pool'):
            for idx, filter_width in enumerate(self.filter_widths):
                with tf.variable_scope('context-%s' % idx):
                    # conv layer
                    filter_shape = [filter_width, self.embedding_size, 1, self.num_filters[idx]]  # [h x w x channel_in x channel_out]
                    W = tf.get_variable(name='kernels', shape=filter_shape, dtype=tf.float32,
                                             initializer=tf.truncated_normal_initializer())
                    b = tf.get_variable(name='biases', shape=[self.num_filters[idx]],
                                             initializer=tf.zeros_initializer())
                    conv = tf.nn.conv2d(self.embedding, W, strides=[1, 1, 1, 1],
                                        padding='VALID', name='conv')
                    z = tf.nn.relu(conv + b, name='relu')
                    # max-pool layer across feature map height: [batch_size x 1 x 1 x channel_out]
                    pool = tf.nn.max_pool(z, ksize=[1, self.max_seq_length-filter_width+1, 1, 1], strides=[1, 1, 1, 1],
                                          padding='VALID', name='pool')
                    pools.append(pool)
        # concatenate multiple filter widths and maps
        num_feature_map = sum(self.num_filters)
        self.pool_out = tf.reshape(tf.concat(pools, axis=3), [-1, num_feature_map], name='pool_out')
        # dropout
        with tf.name_scope('dropout'):
            self.pool_out_drop = tf.nn.dropout(self.pool_out, self.dropout_keep_prob, name='dropout')
        # FCN
        with tf.variable_scope('fc'):
            W = tf.get_variable('weights', shape=[num_feature_map, self.num_classes],
                                initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('biases', [self.num_classes],
                                initializer=tf.zeros_initializer())
            self.logits = tf.matmul(self.pool_out_drop, W) + b
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.targets, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

            self.l2loss = tf.nn.l2_loss(W)


    def _create_loss(self):
        '''
        :return:
        '''
        print('Create loss')
        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logits), name='cross_entropy')
            self.loss = tf.add(self.l2loss * 0.01, cross_entropy, name='total_loss')

    def _create_optimizer(self):
        '''
        :return:
        '''
        print('Create optimizer')
        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.optimizer = tf.train.AdamOptimizer(self.lr)

            grads, vars = zip(*self.optimizer.compute_gradients(self.loss))    # list of tuples (gradient, variable)
            clipped_grads, self.grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
            self.clipped_grads_and_vars = zip(clipped_grads, vars)
            self.train_op = self.optimizer.apply_gradients(self.clipped_grads_and_vars, global_step=self.global_step)

    def _create_summary(self):
        '''
        :return:
        '''
        print('Create summary')
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('grad_norm', self.grad_norm)
            for g,v in self.clipped_grads_and_vars:
                if g is not None:
                    tf.summary.histogram('{}/clipped_grad_hist'.format(v.name), g)
            for weight in tf.trainable_variables():
                tf.summary.histogram('{}/grad'.format(weight.name), weight)
            self.summary_op =  tf.summary.merge_all()

    def build_graph(self):
        self._create_placeholder()
        self._inference()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()
