import requests, json, random, os
from time import sleep

def readjson(fn):
    with open(fn, "r") as file:
        return json.loads(file.read())

n2idx = dict()
idx2n = dict()

for name, idx in readjson('dataset/nameidx.json'):
    n2idx[name] = idx
    idx2n[idx] = name

glob_hist = readjson('dataset/hist.json')

def scrapsub(name, fn):
    if os.path.isfile(fn):
        print(f"{fn} already scrapped. moving on..")
        return

    req = None
    T = 600
    while not req:
        try:
            req = requests.get(f'https://subredditstats.com/api/subredditNameToSubredditsHist?subredditName={name}')
            json.loads(req.text)
        except json.decoder.JSONDecodeError as e:
            print(f'Failed. Trying again after {T} sec')
            sleep(T)
    with open(fn, 'w') as file:
        file.write(req.text)
    print(f"Wrote r/{name} to {fn}. Sleeping it off...")
    sleep(2.3)

for idx in sorted(list(glob_hist.keys()), key=lambda k: -glob_hist[k]):
    name = idx2n[idx]
    print(f'Scrapping r/{name} - {idx}')
    scrapsub(name, f'dataset/subreddits/hist_{idx}.json')


