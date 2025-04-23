
import streamlit as st
import numpy as np
import spacy
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

@st.cache_resource
def load_nlp():
    model_name = "ja_core_news_lg"
    if not spacy.util.is_package(model_name):
        spacy.cli.download(model_name)
    return spacy.load(model_name)

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_keras_model():
    return load_model("best_model20250422.h5")

nlp = load_nlp()
stopwords = nlp.Defaults.stop_words
tokenizer = load_tokenizer()
model = load_keras_model()

# Preprocess
def preprocess(headline):
    doc = nlp(headline)
    tokens = [token.text for token in doc if token.pos_ in {"NOUN", "VERB", "ADJ", "X"} and token.text not in stopwords]
    text = " ".join(tokens)
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=32)
    return padded

# Predictor 
def predict_popularity(headline):
    X = preprocess(headline)
    pred = model.predict(X)
    return "いいね！ヒットの可能性が高いです！" if pred[0][0] >= 0.5 else "ごめんね、ヒットの可能性は低いです…"

# Streamlit UI
st.title("📰 Blueskyバズ予測ツール")

st.markdown("#### 📝 アプリの概要")
st.write("""
これはあなたのBluesky投稿が「バズるかどうか」を予測するアプリです。 
見出しと投稿時間帯を入力するだけで、AIが74%の精度でヒットの可能性を判定してくれます。\\
（モデルは20250422更新）
""")

headline = st.text_input("記事の見出しを入力してください")

if st.button("予測する"):
    result = predict_popularity(headline)
    st.success(result)
