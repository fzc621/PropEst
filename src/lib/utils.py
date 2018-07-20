# -*- coding: utf-8 -*-

import os
import sys
import random

def makedirs(dirname):
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)

def prob_test(prob):
    x = random.random()
    return x <= prob
