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
    saver.restore(sess, config.MODEL_PATH)
    print("Model restored from file: %s" % config.MODEL_PATH)
    for epoch in range(0,10):
        X, Y = mock.get_train_data(batch_size = 10000)
        o1, o2, O1, O2 = sess.run([rn.o1, rn.o2, rn.O1, rn.O2],
                feed_dict={rn.X1:X[0], rn.X2:X[1], rn.O1:Y[0], rn.O2:Y[1]})
        a = (o1 - o2)
        b = (O1 - O2)
        s = np.sign(a * b)
        negative = (s.shape[0] - np.sum(s)) / 2
        print "===== epoch [%d] accuracy [%d/%d = %f] =====" % (
                epoch,
                (s.shape[0] - negative),
                s.shape[0],
                1.0 - (s.shape[0] - np.sum(s)) / 2 / s.shape[0]
        )
