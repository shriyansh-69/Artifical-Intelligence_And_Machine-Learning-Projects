import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Embedding,Dense,Dropout, GlobalAveragePooling1D,Input# pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Model # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import TextVectorization# pyright: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split
import os



@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "spam.csv")

    df = pd.read_csv(csv_path, encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ["label", "message"]
    df.dropna(inplace=True)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


@st.cache_resource
def train_model(df):
    X = df["message"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TextVectorization(
        max_tokens=5000,
        output_sequence_length=100,
        standardize="lower_and_strip_punctuation"
    )

    vectorizer.adapt(X_train)

    X_train_vec = vectorizer(X_train)
    X_test_vec = vectorizer(X_test)

    inputs = Input(shape=(100,))
    x = Embedding(5000, 64)(inputs)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train_vec,
        y_train,
        epochs=5,
        batch_size=32,
        validation_data=(X_test_vec, y_test),
        verbose=0
    )

    _, accuracy = model.evaluate(X_test_vec, y_test, verbose=0)

    return model, vectorizer, accuracy


st.title("Spam Detection App")

df = load_data()
model, vectorizer, accuracy = train_model(df)

st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

user_input = st.text_area("Enter message")

if st.button("Predict"):
    if not user_input or len(user_input.strip()) < 3:
        st.warning("Please enter a valid message")
        st.stop()

    vec = vectorizer([user_input])
    prob = model.predict(vec)[0][0]

    if prob > 0.5:
        st.error(f"Prediction: SPAM\nConfidence: {prob * 100:.2f}%")
    else:
        st.success(f"Prediction: NOT SPAM\nConfidence: {(1 - prob) * 100:.2f}%")
