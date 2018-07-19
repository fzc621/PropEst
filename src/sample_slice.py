# -*- coding: utf-8 -*-

import os
import sys
import random
import timeit
import argparse
from .lib.data_utils import Query, load_query, dump_query
from .lib.utils import makedirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='sample training slices from the whole dataset')
    parser.add_argument('-f', '--fraction', type=float, default=0.01,
        help='training fraction')
    parser.add_argument('-o', '--overlap', type=float,  default=0.2,
        help='training overlap fraction')
    parser.add_argument('train_path', help='train path')
    parser.add_argument('output_dir', help='sampled data dir')
    args = parser.parse_args()

    random.seed()
    start = timeit.default_timer()

    queries = load_query(args.train_path)
    random.shuffle(queries)

    train_size = len(queries)
    slice_size = int(train_size * args.fraction)
    overlap_size = int(slice_size * args.overlap)
    non_overlap_size = slice_size - overlap_size

    slice0 = queries[:slice_size]
    slice1 = slice0[:overlap_size] + queries[slice_size: slice_size + non_overlap_size]
    slice0 = sorted(slice0, key=lambda x:x._qid)
    slice1 = sorted(slice1, key=lambda x:x._qid)

    makedirs(args.output_dir)
    slice0_path = os.path.join(args.output_dir, 'train.slice0.txt')
    slice1_path = os.path.join(args.output_dir, 'train.slice1.txt')

    dump_query(slice0, slice0_path)
    dump_query(slice1, slice1_path)

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
