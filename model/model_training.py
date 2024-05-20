import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from model.data_preparation import load_data, preprocess_data

def train_model(fake_path, real_path):
    df = load_data(fake_path, real_path)
    df = preprocess_data(df)

    x = df["text"]
    y = df["class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    vectorization = TfidfVectorizer()
    xv_train = vectorization.fit_transform(x_train)
    xv_test = vectorization.transform(x_test)

    LR = LogisticRegression()
    LR.fit(xv_train, y_train)

    pred_test = LR.predict(xv_test)
    print(classification_report(y_test, pred_test))

    joblib.dump((vectorization, LR), 'models/fake_news_detector.pkl')
