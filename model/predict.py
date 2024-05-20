
import pandas as pd
import joblib
from model.data_preparation import word_drop

def load_model():
    return joblib.load('models/fake_news_detector.pkl')

def predict_news(model, news):
    vectorization, LR = model
    testing_news = {"text": [news]}
    new_df_test = pd.DataFrame(testing_news)
    new_df_test["text"] = new_df_test["text"].apply(word_drop)
    new_x_test = new_df_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    LR_pred = LR.predict(new_xv_test)
    return "Fake News" if LR_pred[0] == 0 else "Not A Fake News"
