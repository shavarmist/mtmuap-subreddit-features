import os, re, json, json, random
import numpy as np
from math import exp
from scipy.cluster.vq import kmeans2

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

fets = readjson('dataset/features.json')
k = 10
V = np.zeros((n, k))
for i in range(0, n):
    vec = np.array(fets[i2idx[i]])
    V[i,:] = vec / np.maximum(np.linalg.norm(vec), 1e-3)

print("kmeans2...")

n_cluster = 200
means, labels = kmeans2(V, n_cluster, minit='++', seed=0)

print("saving...")

clusters = [[] for i in range(0, n_cluster)]
for i in range(0, n):
    clusters[labels[i]].append(i2idx[i])

with open('dataset/clusters.json', 'w') as file:
    file.write(json.dumps(clusters))

