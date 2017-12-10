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

import math
import random
import numpy as np
import config

def ground_truth_score(feature_vec):
    '''
    Calc score by mocked target weight
    @param feature_vec array: feature vector of a query-doc
    @return float: score of this retrieval
    '''
    score = 0.0
    score += feature_vec[0] * 15
    score += math.pow(feature_vec[1] * 10, 2)
    for i in xrange(len(feature_vec)):
        score += feature_vec[i]
    return score

def labels_for_nonlinear_samples(qd_vec):
    '''
    Calc labels for query-docs
    @param qd_vec array: array of query-doc feature vectors
    @return array: corresponding labels. 1 for top1, -1 for others
    '''
    # sort
    def pairwise_cmp(x, y):
        a = x
        b = y
        if min(x[0], y[0]) == x[0]:
            a = y
            b = x
        coef = 1 if np.array_equal(a, x) else -1
        if a[0] == b[0]:
            return coef * cmp(a[1], b[1])
        elif a[0] == 1.0 or b[0] == 0:
            return coef * 1
        elif a[0] - b[0] <= 0.2:
            # (a[0], b[0]) = (0.7, 0.5), (0.5, 0.3)
            if b[1] - a[1] >= 0.5:
                # (a[1], b[1]) = (0.1, 0.6)
                # not (a[1], b[1]) = (0.6, 1.0)
                return coef * -1
            else:
                return coef * 1
        elif a[1] == 0 and b[1] >= 0.6:
            return coef * -1
        else:
            return coef * 1
        pass
    sorted_qd_vec = sorted(qd_vec, cmp=pairwise_cmp ,reverse=True)
    label_vec = []
    for qd in qd_vec:
        if np.array_equal(qd, sorted_qd_vec[0]):
            label_vec.append(1)
        else:
            label_vec.append(-1)
    pass
    return label_vec

def generate_labeled_data_file(fout, query_count = 100, query_doc_count = config.MOCK_QUERY_DOC_COUNT):
    '''
    Generate mocked data and write to svmlight format file
    '''
    for i in xrange(query_count):
        if config.USE_HIDDEN_LAYER == False:
            # generate linear sample
            for j in xrange(query_doc_count):
                qid = str(i)
                f_vec = np.random.random_sample(config.FEATURE_NUM).round(2)
                label = ground_truth_score(f_vec)

                # FORMAT: label qid:xx 1:x 2:y ..
                rep_str = "" + str(label) + " " + "qid:" + qid
                for k in xrange(len(f_vec)):
                    rep_str += " " + str(k + 1) + ":"
                    rep_str += str(f_vec[k])
                fout.write("%s\n" % (rep_str))
            pass
        else:
            # generate nonlinear sample
            f_arr = []
            for j in xrange(query_doc_count):
                f_vec = np.random.random_sample(config.FEATURE_NUM).round(2)
                if f_vec[0] < 0.3:
                    if np.random.random() < 0.5:
                        f_vec[0] = 0
                if f_vec[1] > 0.6:
                    if np.random.random() > 0.5:
                        f_vec[1] = 1.0
                f_arr.append(f_vec)

            l_arr = labels_for_nonlinear_samples(f_arr)

            # FORMAT: label qid:xx 1:x 2:y ..
            for j in xrange(len(f_arr)):
                rep_str = "" + str(l_arr[j]) + " " + "qid:" + str(i)
                f_vec = f_arr[j]
                for k in xrange(len(f_vec)):
                    rep_str += " " + str(k + 1) + ":"
                    rep_str += str(f_vec[k])
                fout.write("%s\n" % (rep_str))
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
        line = line.encode("utf8")
        line = line.split("#")[0]
        elems = line.split(" ")
        label = float(elems[0])
        qid = elems[1].split(":")[1]
        feature_v = [0.0] * config.FEATURE_NUM
        for i in xrange(2, config.FEATURE_NUM + 2):
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
    if config.USE_TOY_DATA == True:
        fin = open(config.TRAIN_DATA, "w")
        generate_labeled_data_file(fin, 100)
        fin.close()
    #fout = open(config.TRAIN_DATA, "r")
    #data, data_keys = parse_labeled_data_file(fout)
    #fout.close()
    #print "--- parsed pointwise data ---"
    #print data
    #print "--- parsed pairwise data ---"
    #for k, v in data.iteritems():
    #    print "pairs for key [%s]:" % (k)
    #    print calc_query_doc_pairwise_data(v)
