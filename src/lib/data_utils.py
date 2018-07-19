# -*- coding: utf-8 -*-

def load_query(path):
    queries = []
    with open(path, 'r') as fin:
        for line_ in fin:
            line = line_.strip()
            toks = line.split(' ', 2)
            assert len(toks) == 3
            rel = int(toks[0])
            qid = int(toks[1].split(':')[1])
            if len(queries) == 0 or queries[-1].equal_qid(qid) == False:
                queries.append(Query(qid, (rel, toks[2])))
            else:
                queries[-1].append((rel, toks[2]))
    return queries

def dump_query(queries, path):
    queries = []
    with open(path, 'w') as fout:
        for query in queries:
            for doc in query._docs:
                rel, feature = doc
                fout.write('{} qid:{} {}\n'.format(rel, query._qid, feature))

class Query(object):

    def __init__(self, qid, doc):
        self._qid = qid
        self._docs = [doc]

    def append(self, doc):
        self._docs.append(doc)
        return len(self._docs)

    def equal_qid(self, qid):
        return qid == self._qid