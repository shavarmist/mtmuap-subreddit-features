import os, re, json, json, random
import numpy as np

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

i = 0
for idx in readjson('dataset/trainable.json'):
    idx2i[idx] = i
    i2idx[i] = idx
    i += 1

