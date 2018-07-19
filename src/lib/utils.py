# -*- coding: utf-8 -*-

import os
import sys

def makedirs(dirname):
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
