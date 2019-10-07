# coding:utf-8

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from twitterscraper import query_tweets
import datetime
import json
import pickle
import tensorflow as tf


def preprocess(data, tokenizer, maxlen=280):
    return(pad_sequences(tokenizer.texts_to_sequences(data), maxlen=maxlen))


def predict(sentences, graph, emolabels, tokenizer, model, maxlen):
    preds = []
    targets = preprocess(sentences, tokenizer, maxlen=maxlen)
    with graph.as_default():
        for i, ds in enumerate(model.predict(targets)):
            preds.append({
                "sentence": sentences[i],
                "emotions": dict(zip(emolabels, [str(round(100.0*d)) for d in ds]))
            })
    return preds


def load(path):
    model = load_model(path)
    graph = tf.get_default_graph()
    return model, graph


if __name__ == "__main__":
    maxlen = 280
    model, graph = load("sentimentja/model_2018-08-28-15:00.h5")

    with open("sentimentja/tokenizer_cnn_ja.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    emolabels = ["happy", "sad", "disgust", "angry", "fear", "surprise"]

    # acquiring for tweet
    list_of_tweets = query_tweets(
        "lang:ja", begindate=datetime.date(2019, 1, 1), limit=100000, poolsize=64)
    # extract only tweet.text
    list_of_tweets_text = [(tweet.text)
                           for tweet in list_of_tweets if (tweet.is_retweet == 0)]

    text_with_emotion_list = predict(
        list_of_tweets_text, graph, emolabels, tokenizer, model, maxlen)

    emo_count = {
        "happy": 0, "sad": 0, "disgust": 0, "angry": 0, "fear": 0, "surprise": 0
    }
    for text_with_emotion in text_with_emotion_list:
        emotions = text_with_emotion['emotions']
        max_emos = [max_emotions[0] for max_emotions in emotions.items() if max_emotions[1] == max(
            emotions.items(), key=(lambda emotion: float(emotion[1])))[1]]

        # print("-------------")
        # print(text_with_emotion['sentence'])
        # print(list(max_emos))
        # print("=============")

        for max_emo in max_emos:
            emo_count[max_emo] += 1

    print("-------------")
    print(len(text_with_emotion_list))
    print(str(emo_count))
    print("=============")
