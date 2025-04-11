
import streamlit as st
import numpy as np
import spacy
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# NNモデル・Tokenizer・spacy.GiNZAの読み込み
nlp = spacy.load("ja_ginza")
stopwords = nlp.Defaults.stop_words

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("best_model20250408.h5")

# 時間帯エンコード
def encode_hour_period(hour_str):
    mapping = {"morning":0, "noon":1, "afternoon":2, "evening":3, "midnight":4}
    one_hot = np.zeros(5)
    one_hot[mapping[hour_str]] = 1
    return one_hot

# 前処理
def preprocess(headline, hour_period):
    doc = nlp(headline)
    tokens = [token.text for token in doc if token.pos_ in {"NOUN", "VERB", "ADJ", "X"} and token.text not in stopwords]
    text = " ".join(tokens)
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=32)
    hour_encoded = encode_hour_period(hour_period).reshape(1, -1)
    return np.hstack([padded, hour_encoded])

# 予測
def predict_popularity(headline, hour_period):
    X = preprocess(headline, hour_period)
    pred = model.predict(X)
    return "いいね！ヒットの可能性が高いです！" if pred[0][0] >= 0.5 else "ごめんね、ヒットの可能性は低いです…"

# Streamlit UI
st.title("📰 Blueskyバズ予測ツール")

st.markdown("#### 📝 アプリの概要")
st.write("""
これはあなたのBluesky投稿が「バズるかどうか」を予測するアプリです。 
見出しと投稿時間帯を入力するだけで、AIが74%の精度でヒットの可能性を判定してくれます。

7～11時 →　morning
12～14時 → noon
15～19時 → afternoon
20～23時 → evening
24～6時 → midnight
""")

headline = st.text_input("記事の見出しを入力してください")
hour_period = st.selectbox("投稿時間帯を選んでください", ["morning", "noon", "afternoon", "evening", "midnight"])

if st.button("予測する"):
    result = predict_popularity(headline, hour_period)
    st.success(result)
