# -*- coding: utf-8 -*-

import argparse
import math
import os
import pickle
import random
import sys
import timeit
from collections import Counter, defaultdict

import numpy as np
import scipy.optimize as opt

from .lib.data_utils import Query, load_log


def relError(p, p_):
    return np.mean(np.abs((np.array(p) - np.array(p_)) / np.array(p)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='propensity estimation')
    parser.add_argument('--eta', default=1.0, type=float,
                        help='the parameter controls the severity of bias')
    parser.add_argument('-n', default=10, type=int,
        help='number of top positions for which estimates are desired')
    parser.add_argument('-a', '--approach', default='equitation',
                        choices=['naive', 'optimizer', 'chain', 'imp', '_naive', '_chain', 'lsm'])
    parser.add_argument('-m', '--method', default='TNC',
                        choices=['L-BFGS-B', 'TNC', 'SLSQP'])
    parser.add_argument('-l', action='store_true', help='p_k / p_\{M/2\}')
    parser.add_argument('log_dir', help='click log dir')
    parser.add_argument('output_path', help='propensity result path')
    args = parser.parse_args()

    start = timeit.default_timer()

    M = args.n
    l = M // 2 if args.l else 1
    pkl_path = os.path.join(args.log_dir, 'click.pkl')
    if not os.path.exists(pkl_path):
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
                v_ = (1 - delta) / w[(qid, doc_id, rk)]
                for rk_ in range(1, M + 1):
                    if (qid, doc_id) in S[(rk, rk_)]:
                        c.update({(rk, rk_): v})
                        not_c.update({(rk, rk_): v_})
        for q in log1:
            qid = q._qid
            docs = q._docs
            for rk, doc in enumerate(docs, start=1):
                if rk > M:
                    break
                doc_id, delta = doc
                v = delta / w[(qid, doc_id, rk)]
                v_ = (1 - delta) / w[(qid, doc_id, rk)]
                for rk_ in range(1, M + 1):
                    if (qid, doc_id) in S[(rk, rk_)]:
                        c.update({(rk, rk_): v})
                        not_c.update({(rk, rk_): v_})
        with open(pkl_path, 'wb') as fout:
            pickle.dump((c, not_c), fout)
    else:
        with open(pkl_path, 'rb') as fin:
            c, not_c = pickle.load(fin)

    print(c, not_c)
    exit()
    def p_idx(k):
        return (k - 1) * M + k - 1

    def r_idx(k, k_):
        assert k != k_
        if k < k_:
            return (k - 1) * M + k_ - 1
        else:
            return (k_ - 1) * M + k - 1

    a, b = 1e-6, 1-1e-6
    x0 = [random.random() * (b - a) + a] * (M * M)
    bnds = [(a, b)] * (M * M)
    pr = pow(l, -args.eta)
    prop = [pow(k, -args.eta) / pr for k in range(1, M + 1)]
    if args.approach == 'optimizer':
        def likelihood(x):
            r = 0
            for k in range(1, M + 1):
                for k_ in range(1, M + 1):
                    if k != k_:
                        r += c[(k, k_)] * math.log(x[p_idx(k)] * x[r_idx(k, k_)])
                        r += not_c[(k, k_)] * math.log(1 - x[p_idx(k)] * x[r_idx(k, k_)])
            return -r
        best_error = 1
        prop_ = None
        for i in range(30):
            ret = opt.minimize(likelihood, x0, method=args.method, bounds=bnds)
            xm = ret.x
            _prop_ = [xm[p_idx(k)] / xm[p_idx(l)] for k in range(1, M + 1)]
            if relError(prop, _prop_) < best_error :
                best_error = relError(prop, _prop_)
                prop_ = _prop_
    elif args.approach == 'imp':
        N = np.zeros((M,M))
        def likelihood(x):
            r = 0
            for k in range(1, M + 1):
                for k_ in range(1, M + 1):
                    if k != k_:
                        r += c[(k, k_)] * math.log(x[p_idx(k)] * x[r_idx(k, k_)]) / np.sqrt(c[(k, k_)] + not_c[(k, k_)])
                        r += not_c[(k, k_)] * math.log(1 - x[p_idx(k)] * x[r_idx(k, k_)]) / np.sqrt(c[(k, k_)] + not_c[(k, k_)])
            #             N[k-1][k_-1] = 1 / (c[(k, k_)] + not_c[(k, k_)])
            # np.savetxt('N.txt', N)
            # exit()
            return -r
        best_error = 1
        prop_ = None
        for i in range(30):
            ret = opt.minimize(likelihood, x0, method=args.method, bounds=bnds)
            xm = ret.x
            _prop_ = [xm[p_idx(k)] / xm[p_idx(l)] for k in range(1, M + 1)]
            if relError(prop, _prop_) < best_error :
                best_error = relError(prop, _prop_)
                prop_ = _prop_

    elif args.approach == 'naive':
        prop_ = []
        for k in range(1, M + 1):
            if k == l:
                prop_.append(1)
            else:
                prop_.append(c[(k,l)] / c[(l,k)])

    elif args.approach == '_naive':
        def likelihood(x):
            r = 0
            for k in range(1, M + 1):
                for k_ in range(1, M + 1):
                    if k != k_:
                        if k == 1 or k_ == 1:
                            r += c[(k, k_)] * math.log(x[p_idx(k)] * x[r_idx(k, k_)])
                            r += not_c[(k, k_)] * math.log(1 - x[p_idx(k)] * x[r_idx(k, k_)])
            return -r
        ret = opt.minimize(likelihood, x0, method=args.method, bounds=bnds)
        xm = ret.x
        prop_ = [xm[p_idx(k)] / xm[p_idx(l)] for k in range(1, M + 1)]

    elif args.approach == 'chain':
        assert l == 1
        prop_ = []
        for k in range(1, M + 1):
            if k == 1:
                prop_.append(1)
            else:
                a = 1
                b = 1
                for r in range(1, k):
                    a *= c[(r + 1, r)]
                    b *= c[(r, r + 1)]
                prop_.append(a / b)
    elif args.approach == '_chain':
        def likelihood(x):
            r = 0
            for k in range(1, M + 1):
                for k_ in range(1, M + 1):
                    if k != k_:
                        if k == k_ + 1 or k_ == k + 1:
                            r += c[(k, k_)] * math.log(x[p_idx(k)] * x[r_idx(k, k_)])
                            r += not_c[(k, k_)] * math.log(1 - x[p_idx(k)] * x[r_idx(k, k_)])
            return -r
        ret = opt.minimize(likelihood, x0, method=args.method, bounds=bnds)
        xm = ret.x
        prop_ = [xm[p_idx(k)] / xm[p_idx(l)] for k in range(1, M + 1)]
    elif args.approach == 'lsm':
        b = np.zeros((M - 1) * M + 1)
        A = np.zeros(((M - 1) * M + 1, int((M + 1) * M / 2)))
        for k in range(1, M + 1):
            for k_ in range(1, M + 1):
                if k == k_:
                    continue
                row = (M - 1) * (k - 1) + k_ - 1
                if k_ > k:
                    row -= 1

                p_col = int((M - 1) * M / 2 + k - 1)
                if k < k_:
                    r_col = int((2 * M - k) * (k - 1) / 2 + k_ - k - 1)
                else:
                    r_col = int((2 * M - k_) * (k_ - 1) / 2 + k - k_ - 1)
                b[row] = np.log(c[(k,k_)]/(c[(k,k_)] + not_c[(k,k_)]))
                A[row][p_col] = 1
                A[row][r_col] = 1
        A[(M - 1) * M][int((M - 1) * M / 2)] = 1
        b[(M - 1) * M] = 0
        At = np.transpose(A)
        AtA = np.dot(At, A)
        # print(np.linalg.det(AtA))
        logx = np.dot(np.dot(np.linalg.inv(AtA), A.transpose()), b)
        x = np.exp(logx[int((M - 1) * M / 2):])
        prop_ = [x[k] / x[0] for k in range(M)]
    else:
        prop_ = [1] * M

    with open(args.output_path, 'w') as fout:
        for k in range(1, M + 1):
            fout.write('{:.6f}\t{:.6f}\t{:.6f}\n'.format(prop[k - 1], prop_[k - 1], np.abs(1 - prop_[k - 1] / prop[k - 1])))
    print('{} RelError: {:.5f}.'.format(args.approach, relError(prop, prop_)))
    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
