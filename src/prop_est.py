# -*- coding: utf-8 -*-

import os
import sys
import random
import timeit
import argparse
from .lib.data_utils import Query, load_log
from collections import defaultdict, Counter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='propensity estimation')
    parser.add_argument('--eta', default=1.0, type=float,
                        help='the parameter controls the severity of bias')
    parser.add_argument('-n', default=10, type=int,
        help='number of top positions for which estimates are desired')
    parser.add_argument('log_dir', help='click log dir')
    parser.add_argument('output_path', help='propensity result path')
    args = parser.parse_args()

    start = timeit.default_timer()

    M = args.n
    log0_path = os.path.join(args.log_dir, 'log0.txt')
    log1_path = os.path.join(args.log_dir, 'log1.txt')
    log0 = load_log(log0_path)
    log1 = load_log(log1_path)

    S = defaultdict(set)
    for q0, q1 in zip(log0, log1):
        assert q0._qid == q1._qid
        qid = q0._qid
        docs0 = q0._docs
        docs1 = q1._docs
        for rk0, doc0 in enumerate(docs0, start=1):
            if rk0 > M:
                break
            doc_id0, _ = doc0
            for rk1, doc1 in enumerate(docs1, start=1):
                if rk1 > M:
                    break
                if rk1 == rk0:
                    continue
                doc_id1, _ = doc1
                if doc_id1 == doc_id0:
                    S[(rk0, rk1)].add((qid, doc_id0))
                    S[(rk1, rk0)].add((qid, doc_id0))
                    break

    n0 = len(log0)
    n1 = len(log1)
    assert n0 == n1
    w = Counter()
    for q in log0:
        qid = q._qid
        docs = q._docs
        for rk, doc in enumerate(docs, start=1):
            if rk > M:
                break
            doc_id, _ = doc
            w.update({(qid, doc_id, rk):n0})
    for q in log1:
        qid = q._qid
        docs = q._docs
        for rk, doc in enumerate(docs, start=1):
            if rk > M:
                break
            doc_id, _ = doc
            w.update({(qid, doc_id, rk):n1})

    c = Counter()
    not_c = Counter()
    for q in log0:
        qid = q._qid
        docs = q._docs
        for rk, doc in enumerate(docs, start=1):
            if rk > M:
                break
            doc_id, delta = doc
            v = delta / w[(qid, doc_id, rk)]
            for rk_ in range(1, 11):
                if (qid, doc_id) in S[(rk, rk_)]:
                    c.update({(rk, rk_): v})
                    not_c.update({(rk, rk_): 1 - v})
    for q in log1:
        qid = q._qid
        docs = q._docs
        for rk, doc in enumerate(docs, start=1):
            if rk > M:
                break
            doc_id, delta = doc
            v = delta / w[(qid, doc_id, rk)]
            for rk_ in range(1, 11):
                if (qid, doc_id) in S[(rk, rk_)]:
                    c.update({(rk, rk_): v})
                    not_c.update({(rk, rk_): 1 - v})

    with open(args.output_path, 'w') as fout:
        fout.write('p\tp_\n')
        fout.write('{:.3f}\t{:.3f}\n'.format(1, 1))
        for k in range(2, 11):
            prop = pow(k, -1 * args.eta)
            if c[(1,k)]:
                prop_ = c[(k,1)] / c[(1,k)]
            else:
                prop_ = -1
            fout.write('{:.3f}\t{:.3f}\n'.format(prop, prop_))

    end = timeit.default_timer()
    print('Running time: {:.3}s.'.format(end - start))
