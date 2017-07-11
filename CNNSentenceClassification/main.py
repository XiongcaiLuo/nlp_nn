# -*- coding: utf-8 -*-
"""
CNN for text classification
"""
# TODO: add a test phase for interactivate input [eval]

from __future__ import absolute_import
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np

from CNNSentenceClassification import model
from CNNSentenceClassification import data, embedding


flags = tf.app.flags

flags.DEFINE_string('dataset', 'rt-polarity', 'The dataset for training, support rt-polarity and imdb (default: rt-polarity)')
flags.DEFINE_boolean('embedding_static', False, 'True for static embedding layer, set the weights NO trainable')
flags.DEFINE_string('embedding_init_type', 'new',
                    'Initializing type for embed matrix, random|new|pretrained|google_w2c')

flags.DEFINE_integer('embedding_size', 300, 'Dimensionality of embedding vector (default: 300)')
flags.DEFINE_integer('dropout_keep_prob', 0.4, 'Dropout keep probability (default: 0.4)')
flags.DEFINE_string('num_filters', '10,10,10',
                    'Number of filters corresponding to each filter_width (default: 10,10,10)')
flags.DEFINE_string('filter_widths', '4,5,6',
                    'Width of filter or sliding context window (default: 4,5,6)')


flags.DEFINE_integer('batch_size', 100, 'Batch size (default: 50)')
flags.DEFINE_integer('n_epochs', 20, 'Number of training epochs (default: 20)')
flags.DEFINE_float('learning_rate', 0.005, 'Learning rate (default: 0.001)')
flags.DEFINE_float('max_grad_norm', 3.0, 'Maximum gradient norm for clipping gradient (default: 3.0)')
flags.DEFINE_integer('cpt_skip_step', 10, 'Save checkpoints every this many steps (default: 10)')
flags.DEFINE_integer('eval_skip_step', 100, 'Run evaluation on validation data every this many steps. (default: 20)')
flags.DEFINE_boolean('forward_only', False, 'True for test phase, no back propagation (default: False)')

flags.DEFINE_string('cpt_path', './checkpoints',
                    'The directory to write model checkpoints to.')
flags.DEFINE_string('graph_path', './graphs',
                    'The directory to write model summaries to.')

FLAGS = flags.FLAGS



def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(FLAGS.cpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters from previously trained model")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the model")


def single_step(sess, model, X_batch, Y_batch, dropout, forward_only=True):
    """ one step session running
    :param sess:
    :param model:
    :param X_batch:
    :param Y_batch:
    :param dropout: dropout keep prob
    :param forward_only: True for testing or False for training
    :return: [accuracy, summary, loss, grad_norm]
    """

    input_feed = {model.inputs: X_batch, model.targets: Y_batch, model.dropout_keep_prob: dropout}
    if not forward_only:
        output_feed = [model.train_op,
                       model.accuracy,
                       model.summary_op,
                       model.loss]
    else:
        output_feed = [model.accuracy,
                       model.summary_op,
                       model.loss]
    outputs = sess.run(output_feed, feed_dict=input_feed)
    if not forward_only:
        return outputs[1], outputs[2], outputs[3]    # Accuracy, summary, loss
    else:
        return outputs                               # Accuracy, summary, loss


def _eval_test_set(sess, model, test_data, test_writer, step):
    """ Evaluate on the test set.
    :param sess: session
    :param model:
    :param test_data:
    :param test_writer: tf.summary.FileWriter
    :param step: global_step
    :return:
    """
    start = time.time()
    X_test, Y_test = test_data

    accuracy, sum_op, loss  = single_step(sess, model, X_test, Y_test, 1.0, forward_only=True)
    if test_writer:
        test_writer.add_summary(sum_op, step)

    print('Test iter {}: accuracy {:5.2f}, loss {:5.2f}, time {:5.2f}'.format(step, accuracy, loss, time.time() - start))


def train(embedding_static, embedding_init_type):
    """
    :return:
    """
    # load data and vocab_inv dict
    (X_train, y_train), (X_test, y_test), vocab_inv = data.load_data(dataset=FLAGS.dataset)
    # get embedding matrix
    embedding_init = embedding.load_embedding(embedding_init_type, np.vstack((X_train, X_test)),
                                              vocab_inverse=vocab_inv, embedding_dim=FLAGS.embedding_size)

    # parameters with dataset
    vocab_size = len(vocab_inv)
    max_seq_length = X_train.shape[1]
    num_classes = y_train.shape[1]
    # prepare parameters
    num_filters = list(map(int, FLAGS.num_filters.split(',')))
    filter_widths = list(map(int, FLAGS.filter_widths.split(',')))
    # construct graph
    cnn_model = model.CNNTextModel(vocab_size=vocab_size,
                                   max_seq_length=max_seq_length,
                                   num_classes=num_classes,
                                   batch_size=FLAGS.batch_size,
                                   embedding_size=FLAGS.embedding_size,
                                   num_filters=num_filters,
                                   filter_widths=filter_widths,
                                   learning_rate=FLAGS.learning_rate,
                                   max_grad_norm=FLAGS.max_grad_norm,
                                   embedding_static=embedding_static,
                                   embedding_init=embedding_init)
    cnn_model.build_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('Running session')
        train_writer = tf.summary.FileWriter(FLAGS.graph_path+'/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.graph_path+'/test')

        sess.run(tf.global_variables_initializer())
        _check_restore_parameters(sess, saver=saver)

        initial_step = cnn_model.global_step.eval()

        n_batches = int(len(X_train) / FLAGS.batch_size)
        train_batch_gen = data.next_batch(data=(X_train, y_train), batch_size=FLAGS.batch_size)
        total_loss = 0
        for iteration in range(initial_step, n_batches*FLAGS.n_epochs):
            start = time.time()
            X_batch, y_batch = train_batch_gen.next()
            _, summary, loss_batch = single_step(sess, cnn_model, X_batch, y_batch, FLAGS.dropout_keep_prob, forward_only=False)

            train_writer.add_summary(summary, iteration)
            total_loss += loss_batch


            if (iteration+1) % FLAGS.cpt_skip_step == 0:
                print('Batch iter {}: loss {:5.2f}, time {:5.2f}'.format(iteration, total_loss/FLAGS.cpt_skip_step, time.time() - start))
                start = time.time()
                total_loss = 0       # total_loss reset to 0
                saver.save(sess, os.path.join(FLAGS.cpt_path, 'cnn'), global_step=cnn_model.global_step)
            if ((iteration+1) % FLAGS.cpt_skip_step == 0) or (iteration == n_batches*FLAGS.n_epochs - 1):
                # Run evals on development set and print their loss
                _eval_test_set(sess, cnn_model, (X_batch, y_batch), None, iteration)
                _eval_test_set(sess, cnn_model, (X_test, y_test), test_writer, iteration)


        train_writer.close()
        test_writer.close()


def main(_argv):
    # create checkpoints and summaries folder if there isn't one
    if not os.path.exists(FLAGS.cpt_path):
        os.makedirs(FLAGS.cpt_path)

    if not os.path.exists(FLAGS.graph_path+'/train'):
        os.makedirs(FLAGS.graph_path+'/train')

    if not os.path.exists(FLAGS.graph_path+'/train'):
        os.makedirs(FLAGS.graph_path+'/test')
    train(FLAGS.embedding_static, FLAGS.embedding_init_type)


if __name__ == '__main__':
    tf.app.run()
