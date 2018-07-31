# -*- coding: utf-8 -*-

import os
import sys
import timeit
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from .lib.utils import makedirs

def read_prop(path):
    with open(path, 'r') as fin:
        y, y_ = np.loadtxt(fin, unpack=True)
    return y, y_

def _MSE(y, y_):
    return np.mean((y - y_) ** 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval the result')
    parser.add_argument('-k', type=int, help='#Runs')
    parser.add_argument('param_dir', help='parameter directory')
    parser.add_argument('output_dir', help='output dir')
    parser.add_argument('xlabel', nargs='+', help='x label')

    args = parser.parse_args()

    start = timeit.default_timer()

    params = [c for c in os.scandir(args.param_dir)
                if not c.name.startswith('.') and c.is_dir()]
    columns = sorted([int(c.name) for c in params])

    k = args.k
    metric = {}
    for col in columns:
        metric[col] = {}
        for i in range(1, k + 1):
            run_key = '#{}'.format(i)
            est_path = os.path.join(args.param_dir,
                                        '{}/{}/est.txt'.format(col, i))
            y, y_ = read_prop(est_path)
            metric[col][run_key] = _MSE(y, y_)

    metric_df = pd.DataFrame(metric, columns=columns, dtype='float64')
    metric_df.loc['avg'] = metric_df.mean()
    metric_df.loc['std'] = metric_df.std()
    metric_df.to_csv(os.path.join(args.output_dir, 'result.csv'), float_format='%.6f')

    plt.figure()
    plt.errorbar(columns, metric_df.loc['avg'], yerr=metric_df.loc['std'])
    plt.xlabel(' '.join(args.xlabel))
    plt.ylabel('MSE')
    plt.savefig(os.path.join(args.output_dir, 'result.png'))

    end = timeit.default_timer()
    print('Runing time: {:.3f}s.'.format(end - start))
