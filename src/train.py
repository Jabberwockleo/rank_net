#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2017 Wan Li. All Rights Reserved
#
########################################################################

"""
File: ranknet.py
Author: Wan Li
Date: 2017/11/27 10:41:01
"""

import tensorflow as tf
import numpy as np
import ranknet as rn
import mock
import config

fin = open(config.TRAIN_DATA, "w")
mock.generate_labeled_data_file(fin, 500)
fin.close()
fout = open(config.TRAIN_DATA, "r")
train_data = mock.parse_labeled_data_file(fout)
fout.close()

train_data_keys = train_data.keys()
train_data_key_count = len(train_data_keys)

def merge_batch(X, Y, x, y):
    X[0] = X[0] + x[0]
    X[1] = X[1] + x[1]
    Y[0] = Y[0] + y[0]
    Y[1] = Y[1] + y[1]
    return X, Y

saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    query_doc_index = 0
    for epoch in range(0,10000):
        X, Y = [[],[]], [[], []]
        for _ in xrange(config.TRAIN_BATCH_SIZE):
            query_doc_index += 1
            query_doc_index %= len(train_data_keys)
            key = train_data_keys[query_doc_index]
            doc_list = train_data[key]
            x, y = mock.calc_query_doc_pairwise_data(doc_list)
            merge_batch(X, Y, x, y)
        #X, Y = mock.get_train_data(batch_size = 1000)
        X, Y = (np.array(X[0]), np.array(X[1])), (np.array(Y[0]), np.array(Y[1]))
        sess.run(rn.train_op, feed_dict={rn.X1:X[0], rn.X2:X[1], rn.O1:Y[0], rn.O2:Y[1]})
        if epoch % 100 == 0 :
            l_v = sess.run(rn.loss, feed_dict={rn.X1:X[0], rn.X2:X[1], rn.O1:Y[0], rn.O2:Y[1]})
            o12 = sess.run(rn.o12, feed_dict={rn.X1:X[0], rn.X2:X[1], rn.O1:Y[0], rn.O2:Y[1]})
            O12 = sess.run(rn.O12, feed_dict={rn.X1:X[0], rn.X2:X[1], rn.O1:Y[0], rn.O2:Y[1]})
            sign_t = np.sign(o12 * O12)
            negative_count = (sign_t.shape[0] - np.sum(sign_t)) / 2
            print "------ epoch[%d] loss_v[%f] pairwise accuracy [%d/%d = %f] ------ "%(
                epoch,
                l_v,
                (sign_t.shape[0] - negative_count),
                sign_t.shape[0],
                1.0 - (sign_t.shape[0] - np.sum(sign_t)) / 2 / sign_t.shape[0]
        )
    save_path = saver.save(sess, config.MODEL_PATH)
    print("Model saved in file: %s" % save_path)
