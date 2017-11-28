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
