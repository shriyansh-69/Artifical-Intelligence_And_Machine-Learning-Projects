import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

# ---------- LOAD FILES ----------
model = load_model("Chatbot_Model.keras")
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
intents = json.load(open("intent.json"))

# ---------- FUNCTIONS ----------
def clean(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in tokens]

def bag_of_words(sentence):
    sentence_words = clean(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)[0]
    confidence = np.max(res)
    return classes[np.argmax(res)], confidence

def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

# ---------- STREAMLIT UI ----------
st.set_page_config(
    page_title="AI Fitness Chatbot",
    page_icon="💪",
    layout="centered"
)

st.title("💪 AI Fitness Chatbot")
st.caption("Created by Shriyansh Tiwari")

# ---------- SESSION STATE ----------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------- CHAT DISPLAY ----------
for sender, msg in st.session_state.chat:
    if sender == "user":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")

# ---------- USER INPUT ----------
user_input = st.text_input("Ask anything about fitness:")

if st.button("Send") and user_input:
    tag, confidence = predict_class(user_input)

    if confidence < 0.45:
        response = "I'm not fully sure. Can you rephrase or ask something else?"
    else:
        response = get_response(tag)

    st.session_state.chat.append(("user", user_input))
    st.session_state.chat.append(("bot", response))

    st.experimental_rerun()
