from scipy.stats import binom
import os, re, json, json, random

def readjson(fn):
    with open(fn, "r") as file:
        return json.loads(file.read())

n2idx = dict()
idx2n = dict()

for name, idx in readjson('dataset/nameidx.json'):
    n2idx[name] = idx
    idx2n[idx] = name

glob_hist = readjson('dataset/hist.json')

for k, v in sorted(list(glob_hist.items()), key=lambda e: -e[1]):
    print(f"{v} r/{idx2n[k] if k in idx2n else k}")

