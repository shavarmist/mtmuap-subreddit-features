import os, re, json, json, random
import numpy as np
from math import exp

def readjson(fn):
    with open(fn, "r") as file:
        return json.loads(file.read())

n2idx = dict()
idx2n = dict()

for name, idx in readjson('dataset/nameidx.json'):
    n2idx[name] = idx
    idx2n[idx] = name

idx2i = dict()
i2idx = dict()

n = 0
for idx in readjson('dataset/trainable.json'):
    idx2i[idx] = n
    i2idx[n] = idx
    n += 1

relation = readjson('dataset/incomplete.json')

def sigmoid(x):
    if abs(x) > 6:
        return int(x > 0)
    else:
        return 1/(1 + exp(-x))

class Trainer:

    def __init__(self, n, k, edges, nbatch, lr, lam):
        self.features = np.random.normal(loc=0.0, scale=1.0, size=(k, n))
        self.deltas = np.zeros((k, n))
        self.delta_flag = np.zeros(n)
        self.deg = np.zeros(n)
        self.n = n
        self.k = k
        self.edges = edges

        for i, j, _ in edges:
            self.deg[i] += 1
            self.deg[j] += 1

        self.nbatch = nbatch
        self.lr = lr
        self.lam = lam

    def update_delta(self, i, j, s):
        q = s * np.dot(self.features[:,i], self.features[:,j])
        grad = -s * sigmoid(-q) * self.features[:,j] + self.lam / self.deg[i] * self.features[:,i]
        self.deltas[:,i] -= self.lr * grad
        self.delta_flag[i] = 1

    def push_changes(self, i):
        if self.delta_flag[i]:
            self.features[:,i] += self.deltas[:,i]
            self.deltas[:,i] = np.zeros((self.k,))
            self.delta_flag[i] = 0

    def do_batch(self, edges):
        for i, j, s in edges:
            self.update_delta(i, j, s)
            self.update_delta(j, i, s)
        for i, j, _ in edges:
            self.push_changes(i)
            self.push_changes(j)

    def epoch(self):
        print("Running epoch")
        j = 0
        batched_edges = []
        for e in self.edges:
            batched_edges.append(e)
            j += 1
            if j == self.nbatch:
                j = 0
                self.do_batch(batched_edges)
                batched_edges.clear()
        if j != 0:
            self.do_batch(batched_edges)

    def report(self):
        print("Computing error profile...")
        errors = []
        for i, j, s in self.edges:
            q = -s * np.dot(self.features[:,i], self.features[:,j])
            errors.append(sigmoid(q))
        errors.sort()
        print(' '.join([f"{errors[i]:.3f}" for i in range(0, len(errors), int(len(errors)/50))]))

trainer = Trainer(
    n = n,
    k = 10,
    edges = list(map(lambda e: (idx2i[e[0]], idx2i[e[1]], e[2]), relation)),
    nbatch = 1024,
    lr = .1,
    lam = 3
)

for i in range(0, 5):
    trainer.epoch()
    trainer.report()

