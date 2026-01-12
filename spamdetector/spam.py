import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization,Embedding,Dense,Dropout,GlobalAveragePooling1D#pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Sequential#pyright: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split
import os



@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "spam.csv")

    df = pd.read_csv(csv_path, encoding="latin-1")

    # Keep only required columns
    df = df.iloc[:, :2]
    df.columns = ["label", "message"]

    df.dropna(inplace=True)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    return df


@st.cache_resource
def train_model(df):
    x = df["message"].values
    y = df["label"].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    vectorizer = TextVectorization(
        max_tokens=5000,
        output_sequence_length=100,
        standardize="lower_and_strip_punctuation"
    )

    vectorizer.adapt(x_train)

    model = Sequential([
        vectorizer,
        Embedding(5000, 64),
        GlobalAveragePooling1D(),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_data=(x_test, y_test),
        verbose=0
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    return model, accuracy

# InterFace

st.title("Spam Detection App")
st.write("Deep Learning based Spam Classifier using TensorFlow")

df = load_data()
model, accuracy = train_model(df)

st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

user_input = st.text_area("Enter message text")

if st.button("Predict"):
    if not user_input or len(user_input.strip()) < 3:
        st.warning("Please enter a valid message")
        st.stop()

    probability = model.predict([user_input])[0][0]

    if probability > 0.5:
        st.error(f"Prediction: SPAM\nConfidence: {probability * 100:.2f}%")
    else:
        st.success(f"Prediction: NOT SPAM\nConfidence: {(1 - probability) * 100:.2f}%")
