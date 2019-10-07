import pandas as pd
from os.path import join
from sklearn.utils import shuffle

emotions = ["happy", "sad", ["disgust", "disgust2"], "angry", "fear", "surprise"]
dir_path = "gathering/ja_tweets_sentiment"
size = 60000
df = []
for i, es in enumerate(emotions):
    if isinstance(es, list):
        for e in es:
            data = shuffle(pd.read_json(join(dir_path, "{}.json".format(e)))).iloc[:int(size/len(es))]
            data['label'] = i
            df.append(data)
    else:
        data = shuffle(pd.read_json(join(dir_path, "{}.json".format(es)))).iloc[:int(size)]
        data['label'] = i
        df.append(data)

df = pd.concat(df)
df = shuffle(df)
X = df['text']
y = df['label']
df.shape
