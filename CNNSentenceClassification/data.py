# -*- coding: utf-8 -*-
"""
Support two datasets: imdb and rt-polarity
"""
from __future__ import absolute_import
from __future__ import print_function

import re
import numpy as np
from tensorflow.contrib.learn.python.learn import preprocessing
from keras.datasets import imdb
from keras.preprocessing import sequence


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    Taken from https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def process_raw_data(filenames, vocab_save_path):
    """
    :param filenames: filename path list
    :return:
    """
    pos_path, neg_path = filenames
    x_text, y = load_data_and_labels(pos_path, neg_path)

    # build vocab and sentence to id
    max_seq_length = max([len(x.split()) for x in x_text])
    vocab_processor = preprocessing.VocabularyProcessor(max_seq_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # random shuffle data and split train test dataset
    np.random.seed(113)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # np.random.shuffle(x)
    # np.random.shuffle(y)

    dev_sample_index = int(0.9 * float(len(y)))
    x_train, x_test = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_test = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))
    vocab_processor.save(vocab_save_path)

    vocab_inv = dict((id, token) for token, id in vocab_processor.vocabulary_._mapping.iteritems())

    return (x_train, y_train), (x_test, y_test), vocab_inv


def next_batch(data=None, batch_size=50, shuffle=True):
    """ Return the next `batch_size` examples from this data set."""
    x, y = data
    num_examples = x.shape[0]

    # Shuffle the data
    if shuffle:
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        x = x[perm]
        y = y[perm]

    # fetch data
    start = 0
    while True:
        # Go to the next epoch
        if start + batch_size > num_examples:
            # Get the rest examples in current epoch
            rest_num_examples = num_examples - start
            x_rest_part = x[start:num_examples]
            y_rest_part = y[start:num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(num_examples)
                np.random.shuffle(perm)
                x = x[perm]
                y = y[perm]
            # Start next epoch
            start = 0
            end = batch_size - rest_num_examples
            x_new_part = x[start:end]
            y_new_part = y[start:end]
            yield np.concatenate((x_rest_part, x_new_part), axis=0), \
                  np.concatenate((y_rest_part, y_new_part), axis=0)
            # update the new start index
            start = end
        else:
            end = start + batch_size
            yield x[start:end], y[start:end]
            # update the new start index
            start = end


def load_data(dataset='rt-polarity'):
    """
    prepare datasets
    :param dataset: dataset name
    :return:
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`
        vocab_inv (Dict): {int: str}
    """
    if dataset == 'imdb':
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=5000)
        x_train = sequence.pad_sequences(x_train, maxlen=400)
        x_test = sequence.pad_sequences(x_test, maxlen=400)

        print('train shape {0}, test shape {1}'.format(x_train.shape, x_test.shape))

        vocab = imdb.get_word_index()
        vocab_inv = dict((v, k) for k, v in vocab.items())
        vocab_inv[0] = u'<PAD/>'
        return (x_train, y_train), (x_test, y_test), vocab_inv
    else:
        pos_path = './data/rt-polarity.pos'
        neg_path = './data/rt-polarity.neg'

        return process_raw_data([neg_path, pos_path], './data/rt-polarity.vocab')



if __name__ == '__main__':
    train_data, test_data, vocab_inv = load_data(dataset='')
    g = next_batch(train_data, batch_size=50)
    for i in range(5):
        x, y = g.next()
        print(x.shape)
        print(y.shape)
