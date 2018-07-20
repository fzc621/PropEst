# -*- coding: utf-8 -*-

import os
import sys
import random
import timeit
import argparse
from .lib.data_utils import Query
from .lib.utils import prob_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simulate the clicks')
    parser.add_argument('--eta', default=1.0, type=float,
                        help='the parameter controls the severity of bias')
    parser.add_argument('--epsilon_p', default=1.0, type=float,
                        help='the prob of users click on a relevant result')
    parser.add_argument('--epsilon_n', default=0.1, type=float,
                        help='the prob of users click on a irrelevant result')
    parser.add_argument('data_path', help='data path')
    parser.add_argument('score_path', help='score path')
    parser.add_argument('log_path', help='click log path')

    args = parser.parse_args()

    ep = args.epsilon_p
    en = args.epsilon_n

    start = timeit.default_timer()

    pos_pr = {rk : pow(rk, -1 * args.eta) for rk in range(1, 1000)}
    queries = []
    with open(args.data_path, 'r') as fin1, open(args.score_path, 'r') as fin2:
        for line1, line2 in zip(fin1, fin2):
            line1, line2 = line1.strip(), line2.strip()
            toks = line1.split(' ', 2)
            assert len(toks) == 3
            rel = int(toks[0])
            qid = int(toks[1].split(':')[1])
            score = float(line2)
            if not queries or not queries[-1].equal_qid(qid):
                queries.append(Query(qid, (0, score, rel)))
            else:
                doc_id = len(queries[-1]._docs)
                queries[-1].append((doc_id, score, rel))

    with open(args.log_path, 'w') as fout:
        for query in queries:
            qid = query._qid
            docs = sorted(query._docs, key=lambda x: x[1], reverse=True)
            for rk, sr in enumerate(docs, start=1):
                pr = pos_pr[rk]
                doc_id, _, rel = sr
                clicked = False
                if prob_test(pr):
                    if rel:
                        if prob_test(ep):
                            clicked = True
                    else:
                        if prob_test(en):
                            clicked = True
                if clicked:
                    fout.write('{} qid:{} {}\n'.format(1, qid, doc_id))
                else:
                    fout.write('{} qid:{} {}\n'.format(0, qid, doc_id))

    end = timeit.default_timer()
    print('Running time: {:.3f}s.'.format(end - start))
