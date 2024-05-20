import pandas as pd
import re
import string

def load_data(fake_path, real_path):
    df_fake = pd.read_csv(fake_path)
    df_real = pd.read_csv(real_path)

    df_fake["class"] = 0
    df_real["class"] = 1

    df_f = df_fake.tail(10)
    for i in range(23480, 23470, -1):
        df_fake.drop([i], axis=0, inplace=True)
    df_t = df_real.tail(10)
    for i in range(21416, 21406, -1):
        df_real.drop([i], axis=0, inplace=True)

    df_manual = pd.concat([df_f, df_t])
    df_manual.to_csv("manual_testing.csv")

    df = pd.concat([df_real, df_fake], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def preprocess_data(df):
    df = df.drop(["date", "title", "subject"], axis=1)
    df = df.sample(frac=1)
    df["text"] = df["text"].apply(word_drop)
    return df

def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', ' ', text)
    text = re.sub("\\W", " ", text)
    text = re.sub("https?://\S+|www\.\S+", " ", text)
    text = re.sub('<.*?>+', " ", text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\W*\d\W*', '', text)
    return text
