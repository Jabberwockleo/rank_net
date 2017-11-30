#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2017 Wan Li. All Rights Reserved
#
########################################################################

"""
File: config.py
Author: Wan Li
Date: 2017/11/27 10:41:01
"""

import random
import numpy as np
import config

def ground_truth_score(feature_vec):
    score = 0.0
    score += feature_vec[0] * 9
    for i in xrange(len(feature_vec)):
        score += feature_vec[i]
    return score

def generate_labeled_data(query = "", fin = None, doc_count = 4):
    for i in xrange(doc_count):
        qid = query
        f_vec = np.random.random_sample(config.FEATURE_NUM).round(2)
        label = ground_truth_score(f_vec)

        rep_str = "" + str(label) + " " + "qid:" + qid
        for i in xrange(len(f_vec)):
            rep_str += " " + str(i + 1) + ":"
            rep_str += str(f_vec[i])
        fin.write("%s\n" % (rep_str))
    pass

def get_pairwise_data():
    pass

def get_train_data(batch_size = 100):
    X1, X2 = [],[]
    Y1, Y2 = [],[]
    for i in range(0, batch_size):
        x1 = []
        x2 = []
        o1 = 0.0
        o2 = 0.0
        for j in range(0, config.FEATURE_NUM):
            r1 = random.random()
            r2 = random.random()
            x1.append(r1)
            x2.append(r2)
            mu = 10.0
            if j >= 1 : mu = 1.0
            o1 += r1 * mu
            o2 += r2 * mu
        X1.append(x1)
        Y1.append([o1])
        X2.append(x2)
        Y2.append([o2])

    return  ((np.array(X1), np.array(X2)), (np.array(Y1), np.array(Y2)))

if __name__ == "__main__":
    fin = open(config.SAVE_TRAIN_DATA, "w")
    for i in xrange(10):
        generate_labeled_data("Q" + str(i), fin)
