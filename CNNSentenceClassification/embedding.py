# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function

import os, pdb
import numpy as np
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
from CNNSentenceClassification.data import load_data

model_dir = '../pretrained_models'


def train_word2vec(sentences, vocab_inverse, embedding_dim, min_count=1,
                   window=10, model_name=None):
    """
    :param sentences: num_examples x max_seq_length matrix
    :param vocab_inverse: vocabulary inverse dict {id: token}
    :param embedding_dim:
    :param min_count:
    :param window:
    :return: embedding word vectors
    """
    if model_name is None:
        model_name = '{:d}dim_{:d}mincount_{:d}window'.format(
            embedding_dim, min_count, window)

    print('Training word2vec model using gensim...')
    corpus = [[vocab_inverse[id] for id in seq] for seq in sentences]

    model = word2vec.Word2Vec(size=embedding_dim, min_count=min_count, window=window, sample=1e-3)
    model.build_vocab(sentences=corpus)
    model.train(sentences=corpus, total_examples=model.corpus_count, epochs=model.iter)

    model.init_sims(replace=True)
    # Saving the model
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    print('Saving word2vec model \'%s\'' % model_name)
    model.save(model_dir+'/'+model_name)

    print(model.doesnt_match(words=['breakfast', 'lunch', 'dinner', 'hello']))
    print (model.most_similar(positive=['queen'], negative=['king']))
    # output
    word_vectors = model.wv
    del model
    return word_vectors


def load_embedding(embedding_init_type, sentences, vocab_inverse, embedding_dim=300,
                          min_count=1, window=10, model_name=None):
    """
    Load (may need training) embedding weights
    :param embedding_init_type: str, new|google_w2v|pretrained
    :param sentences: num_examples x max_seq_length matrix
    :param vocab_inverse: vocabulary inverse dict {id: token}
    :param embedding_dim:
    :param min_count:
    :param window:
    :param model_name:
    :return: numpy matrix, vocab_size x embedding_dim
    """
    if embedding_init_type == 'new':
        word_vectors = train_word2vec(sentences, vocab_inverse, embedding_dim, min_count, window)
    elif embedding_init_type == 'google_w2v':
        google_word2vec_path = '../pretrained_models/GoogleNews-vectors-negative300.bin'
        word_vectors = KeyedVectors.load_word2vec_format(google_word2vec_path, binary=True)
    elif embedding_init_type == 'pretrained' and (model_name is not None) and os.path.exists(model_dir+'/'+model_name):
        model = word2vec.Word2Vec.load(model_dir+'/'+model_name)
        word_vectors = model.wv
        print(model.doesnt_match(words=['breakfast', 'lunch', 'dinner', 'hello']))
        print (model.most_similar(positive=['queen'], negative=['king']))
    # output
        del model
    else:
        return None


    # add unknown word vectors using random
    # matrix [vocab_size x embedding_dim]
    vocab_size = len(vocab_inverse)
    words_vocab_order = [vocab_inverse[i] for i in range(vocab_size)]
    vectors_vocab_order = [word_vectors[token] if token in word_vectors else
                           np.random.uniform(-0.25, 0.25, embedding_dim)
                           for token in words_vocab_order]

    embedding_weights = np.vstack(tup=tuple(vectors_vocab_order))
    return embedding_weights


def test_new(dataset='rt-polarity'):
    # load data and vocab_inv dict
    (X_train, y_train), (X_test, y_test), vocab_inv = load_data(dataset)
    embed = load_embedding('new', np.vstack(tup=(X_train, X_test)), vocab_inv, embedding_dim=50)
    print(embed.shape)
    #print(embed[:2])


def test_pretrained():
    embed = load_embedding('new', [[1, 2], [3, 4]],
                   vocab_inverse={0: '<PAD/>', 1: 'hello', 2: 'world', 3:  'once', 4:  'more'},
                   model_name='50features_1minwords_10context')
    print(embed.shape)
    #print(embed)


if __name__ == '__main__':
    test_new('imdb')
