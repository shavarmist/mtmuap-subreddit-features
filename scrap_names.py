import requests, json
from time import sleep

ghist = None
with open('dataset/hist.json', 'r') as file:
    ghist = json.loads(file.read())

nameidxpairs = []

def process_buffer(buff):
    global nameidxpairs
    data = None
    T = 4
    while True:
        try:
            req = requests.post('https://subredditstats.com/api/specificSubredditIdsToNames', json = {'subredditIds':buff})
            data = json.loads(req.text)
        except json.decoder.JSONDecodeError as e:
            print(f'Failed. Trying again after {T} sec')
            sleep(T)
            T *= 2
        if data:
            break
    for i in range(len(buff)):
        nameidxpairs.append([data[i], buff[i]])

buff = []
for k in ghist.keys():
    buff.append(k)
    if len(buff) == 950:
        print(f'Fetching {len(nameidxpairs)} - {len(nameidxpairs)+len(buff)} batch')
        process_buffer(buff)
        sleep(2)
        buff = []
process_buffer(buff)

with open('dataset/nameidx.json','w') as file:
    file.write(json.dumps(nameidxpairs))

