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
import ranknet as rn
import mock

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for epoch in range(0,10000):
        X, Y = mock.get_train_data()
        sess.run(rn.train_op, feed_dict={rn.X1:X[0], rn.X2:X[1], rn.O1:Y[0], rn.O2:Y[1]})
        if epoch % 10 == 0 :
            l_v = sess.run(rn.loss, feed_dict={rn.X1:X[0], rn.X2:X[1], rn.O1:Y[0], rn.O2:Y[1]})
            h_o12_v = sess.run(rn.o12, feed_dict={rn.X1:X[0], rn.X2:X[1], rn.O1:Y[0], rn.O2:Y[1]})
            o12_v = sess.run(rn.O12, feed_dict={rn.X1:X[0], rn.X2:X[1], rn.O1:Y[0], rn.O2:Y[1]})
            print "------ epoch[%d] loss_v[%f] ------ "%(epoch, l_v)
            for k in range(0, len(o12_v)):
                print "k[%d] o12_v[%f] h_o12_v[%f]"%(k, o12_v[k], h_o12_v[k])
