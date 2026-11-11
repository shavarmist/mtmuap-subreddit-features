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

def shuffled(arr):
    arr2 = arr[:]
    random.shuffle(arr2)
    return arr2

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
        for e in shuffled(self.edges):
            batched_edges.append(e)
            j += 1
            if j == self.nbatch:
                j = 0
                self.do_batch(batched_edges)
                batched_edges.clear()
        if j != 0:
            self.do_batch(batched_edges)

    def report(self, edges = None):
        print("Computing error profile...")
        errors = []
        if edges == None:
            edges = self.edges

        for i, j, s in edges:
            q = -s * np.dot(self.features[:,i], self.features[:,j])
            errors.append(sigmoid(q))
        errors.sort()
        n_points = 50
        table = []
        for i in range(0, n_points):
            table.append(f'{100*(i+1)//(n_points+2)}: {errors[len(errors)*(i+1)//(n_points+2)]:.3f}'.rjust(12))

        for r in range(0, len(table)//5):
            print(' '.join(table[5*r:min(5*r+5,len(table))]))

edges = list(map(lambda e: (idx2i[e[0]], idx2i[e[1]], e[2]), relation))
train_edges = edges[200:]
test_edges = edges[:200]

trainer = Trainer(
    n = n,
    k = 10,
    edges = train_edges,
    nbatch = 1024,
    lr = .1,
    lam = 3
)
def train_model():
    n_epoch = 1
    while True:
        try:
            print(f"=== EPOCH {n_epoch} ===")
            trainer.epoch()
            print("=== TRAIN EDGES ===")
            trainer.report()
            print("=== TEST EDGES ===")
            trainer.report(test_edges)
            n_epoch += 1
        except KeyboardInterrupt:
            print("Choices: 1=save 2=continue")
            if input() == '1':
                break
    
    print("Training finished. saving...")
    
    fets = dict()
    
    for i in range(0, n):
        fets[i2idx[i]] = list(trainer.features[:,i])
    
    with open('dataset/features.json', 'w') as file:
        file.write(json.dumps(fets))

def grade_model():
    for k, v in readjson('dataset/features.json').items():
        trainer.features[:,idx2i[k]] = np.array(v)
    print("=== TRAIN EDGES ===")
    trainer.report()
    print("=== TEST EDGES ===")
    trainer.report(test_edges)

grade_model()

