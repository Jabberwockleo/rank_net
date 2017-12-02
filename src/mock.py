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
    '''
    Calc score by mocked target weight
    '''
    score = 0.0
    score += feature_vec[0] * 9
    for i in xrange(len(feature_vec)):
        score += feature_vec[i]
    return score

def generate_labeled_data_file(fout, query_count = 100, query_doc_count = config.MOCK_QUERY_DOC_COUNT):
    '''
    Generate mocked data and write to svmlight format file
    '''
    for i in xrange(query_count):
        for j in xrange(query_doc_count):
            qid = "Q" + str(i)
            f_vec = np.random.random_sample(config.FEATURE_NUM).round(2)
            label = ground_truth_score(f_vec)

            # FORMAT: label qid:xx 1:x 2:y ..
            rep_str = "" + str(label) + " " + "qid:" + qid
            for k in xrange(len(f_vec)):
                rep_str += " " + str(k + 1) + ":"
                rep_str += str(f_vec[k])
            fout.write("%s\n" % (rep_str))
        pass
    pass

def parse_labeled_data_file(fin):
    '''
    Read labefinled data from file (in standard svmlight format)
    @retuen dict[qid]feature_vec, list[qid]
    '''
    data = {}
    keys = []
    last_key = ""
    for line in fin:
        line = line.split("#")[0]
        elems = line.split(" ")
        label = float(elems[0])
        qid = elems[1].split(":")[1]
        feature_v = [0.0] * config.FEATURE_NUM
        for i in xrange(2, len(elems)):
            subelems = elems[i].split(":")
            if len(subelems) < 2:
                continue
            index = int(subelems[0]) - 1
            feature_v[index] = float(subelems[1])
        if qid in data:
            data[qid].append([label] + feature_v)
        else:
            data[qid] = [[label] + feature_v]
        if last_key != qid:
            last_key = qid
            keys.append(qid)

    return data, keys

def calc_query_doc_pairwise_data(doc_list):
    '''
    Calc required sample pairs from one retrival
    e.g. if doc_list contains A, B, C and A > B > C
         pairs are generated as A > B, A > C, B > C
    @param doc_list
        list of list of: [score, f1, f2 , ..., fn]
    @return [X1, X2], [Y1, Y2]
        X1.shape = X2.shape = (None, config.FEATURE_NUM)
        Y1.shape = Y2.shape = (None, 1)
    '''
    X1 = []
    X2 = []
    Y1 = []
    Y2 = []
    sorted_doc_list = sorted(doc_list, cmp=lambda x, y: cmp(y[0], x[0]))
    for i in xrange(len(sorted_doc_list)):
        for j in xrange(i + 1, len(sorted_doc_list), 1):
            X1.append(sorted_doc_list[i][1:])
            Y1.append(sorted_doc_list[i][0:1])
            X2.append(sorted_doc_list[j][1:])
            Y2.append(sorted_doc_list[j][0:1])
    return [X1, X2], [Y1, Y2]

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
    print "=== Unit Test ==="
    fin = open(config.TRAIN_DATA, "w")
    generate_labeled_data_file(fin, 3)
    fin.close()
    fout = open(config.TRAIN_DATA, "r")
    data, data_keys = parse_labeled_data_file(fout)
    fout.close()
    print "--- parsed pointwise data ---"
    print data
    print "--- parsed pairwise data ---"
    for k, v in data.iteritems():
        print "pairs for key [%s]:" % (k)
        print calc_query_doc_pairwise_data(v)
