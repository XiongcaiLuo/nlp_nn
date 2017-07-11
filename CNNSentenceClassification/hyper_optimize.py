# -*- coding: utf-8 -*-
"""

"""
import hyperopt
import numpy as np
import tensorflow as tf

from CNNSentenceClassification import model
from CNNSentenceClassification import data, embedding
from CNNSentenceClassification import main


# TODO: wrap as tf.contrib.learn.estimator


def objective(hp_space):
    # process tuning parameters
    params = {'dataset': 'rt-polarity',
              'embedding_init_type': 'new',
              'embedding_static': False,
              'embedding_size': 300,
              'num_filters': '10,10,10',
              'filter_widths': '3,4,5',
              'batch_size': 50,
              'n_epochs': 10,
              'learning_rate': 0.005,
              'max_grad_norm': 3.0,
              'dropout_keep_prob': 0.5}
    for key in hp_space:
        if key in params:
            params[key] = hp_space[key]


    # load data and vocab_inv dict
    (X_train, y_train), (X_test, y_test), vocab_inv = data.load_data(dataset=params['dataset'])
    # get embedding matrix
    embedding_init = embedding.load_embedding(params['embedding_init_type'], np.vstack((X_train, X_test)),
                                              vocab_inverse=vocab_inv, embedding_dim=params['embedding_size'])

    # parameters with dataset
    vocab_size = len(vocab_inv)
    max_seq_length = X_train.shape[1]
    num_classes = y_train.shape[1]
    # prepare parameters
    num_filters = list(map(int, params['num_filters'].split(',')))
    filter_widths = list(map(int, params['filter_widths'].split(',')))

    with tf.Graph().as_default():
        # construct graph
        cnn_model = model.CNNTextModel(vocab_size=vocab_size,
                                       max_seq_length=max_seq_length,
                                       num_classes=num_classes,
                                       batch_size=params['batch_size'],
                                       embedding_size=params['embedding_size'],
                                       num_filters=num_filters,
                                       filter_widths=filter_widths,
                                       learning_rate=params['learning_rate'],
                                       max_grad_norm=params['max_grad_norm'],
                                       embedding_static=params['embedding_static'],
                                       embedding_init=embedding_init)
        cnn_model.build_graph()


        with tf.Session() as sess:
            print('Running session')

            sess.run(tf.global_variables_initializer())
            initial_step = cnn_model.global_step.eval()

            n_batches = int(len(X_train) / params['batch_size'])
            train_batch_gen = data.next_batch(data=(X_train, y_train), batch_size=params['batch_size'])
            total_loss = 0
            for iteration in range(initial_step, n_batches*params['n_epochs']):
                X_batch, Y_batch = train_batch_gen.next()
                _, summary, loss_batch = main.single_step(sess, cnn_model, X_batch, Y_batch, params['dropout_keep_prob'], forward_only=False)

            # Run evals on development set and print their loss
            acc = cnn_model.accuracy.eval(feed_dict={cnn_model.inputs: X_test, cnn_model.targets: y_test, cnn_model.dropout_keep_prob: 1.0})

    return {'loss': -acc, 'status': hyperopt.STATUS_OK}


def optimize():
    space = {
        'dataset': hyperopt.hp.choice('dataset', ['rt-polarity']),
        'filter_widths': hyperopt.hp.choice('filter_widths', ['2,3,4', '3,4,5', '4,5,6']),
        'num_filters': hyperopt.hp.choice('num_filters', ['10,10,10']),
        'learning_rate': hyperopt.hp.loguniform('learning_rate', -3, 0),
        'dropout_keep_prob': hyperopt.hp.uniform('dropout_keep_prob', 0.4, 0.8)
    }


    best_model = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, max_evals=5)

    print(best_model)
    print(hyperopt.space_eval(space, best_model))
#

if __name__ == '__main__':
    optimize()
