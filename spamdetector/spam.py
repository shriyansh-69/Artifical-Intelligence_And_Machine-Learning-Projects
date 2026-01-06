import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------------
# Load Dataset (Cached)
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

# -----------------------------------
# Train Model (Cached)
# -----------------------------------
@st.cache_resource
def train_model(df):
    X = df['message']
    y = df['label']

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=3000
    )
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return vectorizer, model, accuracy

# -----------------------------------
# App Start
# -----------------------------------
st.title("📧 Spam Detection App")
st.write("Check whether a message is **Spam** or **Not Spam**")

df = load_data()
vectorizer, model, accuracy = train_model(df)

# -----------------------------------
# User Input
# -----------------------------------
user_input = st.text_area("Enter Message")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    elif len(user_input.split()) < 3:
        st.warning("Please enter a longer message")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        probability = model.predict_proba(input_vec)[0]

        if prediction == 1:
            st.error(f"🚨 SPAM\nConfidence: {probability[1]*100:.2f}%")
        else:
            st.success(f"✅ NOT SPAM\nConfidence: {probability[0]*100:.2f}%")

# -----------------------------------
# Accuracy Display
# -----------------------------------
st.write(f"### Model Accuracy: {accuracy*100:.2f}%")
