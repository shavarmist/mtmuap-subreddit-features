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
glob_sum = sum(glob_hist.values())

populars = sorted(list(glob_hist.keys()), key=lambda k: -glob_hist[k])

inconsistent = 0

def add_edge(relation, a, b, c):
    global inconsistent
    a, b = min(a, b), max(a, b)
    if (a, b, -c) in relation:
        inconsistent += 1
        relation.remove((a, b, -c))
    else:
        relation.add((a, b, c))

def observe_relations():
    global glob_sum, glob_hist, n2idx, idx2n, populars

    relation = set()

    for fn in os.listdir('dataset/subreddits'):
        idx = re.search('hist_([0-9]+).json', fn, re.IGNORECASE).group(1)

        part_hist = readjson('dataset/subreddits/'+fn)
        part_sum = sum(part_hist.values())


        def process_jdx(jdx, cnt):
            p = 1
            glob_q = glob_hist[jdx] / glob_sum
            part_q = cnt / part_sum

            if glob_q != part_q:
                if part_q > glob_q:
                    p = binom.sf(cnt - 1, part_sum, glob_q)
                else:
                    p = binom.sf(part_sum - cnt - 1, part_sum, 1-glob_q)

            if p < .01:
                sg = (1 if part_q > glob_q else -1)
                add_edge(relation, idx, jdx, sg)
                return True
            else:
                return False

        work_load = 0

        for jdx, cnt in part_hist.items():
            work_load += 1
            process_jdx(jdx, cnt)

        for jdx in populars:
            work_load += 1
            if jdx not in part_hist:
                if not process_jdx(jdx, 0):
                    break # svi manje popularni subredditi ce takodjer bit neznacajni
        
        print(f"Worked on {idx2n[idx]}, Work load: {work_load}")

    print(f"Total relations recorded: {len(relation)}")
    print(f"Inconsistencies: {inconsistent}")

    with open('dataset/incomplete.json', 'w') as file:
        file.write(json.dumps(list(relation)))

observe_relations()

# relation = readjson('dataset/incomplete.json')

