import json
from itertools import islice

def readjson(fn):
    with open(fn, "r") as file:
        return json.loads(file.read())

n2idx = dict()
idx2n = dict()

for name, idx in readjson('dataset/nameidx.json'):
    n2idx[name] = idx
    idx2n[idx] = name

def cgroups(nodes, edges):
    par, siz = dict(), dict()
    for k in nodes:
        par[k] = k
        siz[k] = 1

    def getpar(x):
        p = par[x]
        while par[p] != p:
            p = par[p]
        p_ = x
        while p_ != p:
            p_, par[p_] = par[p_], p
        return p

    for e in edges:
        p1, p2 = map(getpar, [e[0], e[1]])
        if p1 == p2:
            continue
        if siz[p1] < siz[p2]:
            p1, p2 = p2, p1
        par[p2] = p1
        siz[p1] += siz[p2]

    tribes = dict()
    for k in nodes:
        p = getpar(k)
        if not p in tribes:
            tribes[p] = set()
        tribes[p].add(k)

    return tribes.values()

netw = readjson('dataset/incomplete.json')

S = list(cgroups(idx2n.keys(), netw))
S_ = list(filter(lambda G: len(G) != 1, S))

print("Total number of clusters of information: ", len(S))
print("Number of non-trivial clusters of information: ", len(S_))

assert len(S_) == 1

print("From now on resuming only with the unique connected non-trivial cluster")
C = S_[0]

print(f"Observing {len(C)} nodes")

tribe = dict()
tribe_size = dict()
i = 0
for friends in cgroups(idx2n.keys(), filter(lambda e: e[2] == 1, netw)):
    tribe_size[i] = len(friends)
    for k in friends:
        tribe[k] = i
    i += 1

print(f"Found at most {i} tribes.")

print("Top 100 tribes: ", sorted(list(tribe_size.values()), key=lambda x: -x)[:100])

inconsistent = 0
for e in filter(lambda e: e[2] == -1, netw):
    if tribe[e[0]] == tribe[e[1]]:
        inconsistent += 1

print(f"Found {inconsistent} inconsistencies among {len(netw)} recorded edges.")

with open('dataset/trainable.json', 'w') as file:
    file.write(json.dumps(list(C)))

print("# 1 u 100 error rate je dovoljan na primjeru od 33k cvorova da stvori pozitivne puteve izmedu 90% cvorova")

