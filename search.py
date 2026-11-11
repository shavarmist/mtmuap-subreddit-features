import os, re, json, json, random
import numpy as np
from math import exp
from scipy.cluster.vq import kmeans2
import sys

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
clusters = readjson('dataset/clusters.json')

if len(sys.argv) < 2:
    print("Usage: python search.py <subreddit without r/>")
    sys.exit(1)

subreddit = sys.argv[1]
subidx = n2idx[subreddit]
cluster = None
for cl in clusters:
    if subidx in cl:
        cluster = cl
        break
sorted_cluster = sorted(cluster, key=lambda idx: -np.dot(fets[idx], fets[subidx])/np.linalg.norm(fets[idx]))
print('\n'.join(map(lambda idx: idx2n[idx], sorted_cluster)))


