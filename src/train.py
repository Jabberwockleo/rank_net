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

saver = tf.train.Saver()

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(0,10000):
        X, Y = mock.get_train_data(batch_size = 1000)
        sess.run(rn.train_op, feed_dict={rn.X1:X[0], rn.X2:X[1], rn.O1:Y[0], rn.O2:Y[1]})
        if epoch % 100 == 0 :
            l_v = sess.run(rn.loss, feed_dict={rn.X1:X[0], rn.X2:X[1], rn.O1:Y[0], rn.O2:Y[1]})
            o12 = sess.run(rn.o12, feed_dict={rn.X1:X[0], rn.X2:X[1], rn.O1:Y[0], rn.O2:Y[1]})
            O12 = sess.run(rn.O12, feed_dict={rn.X1:X[0], rn.X2:X[1], rn.O1:Y[0], rn.O2:Y[1]})
            sign_t = np.sign(o12 * O12)
            negative_count = (sign_t.shape[0] - np.sum(sign_t)) / 2
            print "------ epoch[%d] loss_v[%f] accuracy [%d/%d = %f] ------ "%(
                epoch,
                l_v,
                (sign_t.shape[0] - negative_count),
                sign_t.shape[0],
                1.0 - (sign_t.shape[0] - np.sum(sign_t)) / 2 / sign_t.shape[0]
        )
    save_path = saver.save(sess, config.MODEL_PATH)
    print("Model saved in file: %s" % save_path)
