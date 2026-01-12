import pandas as pd 
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, Dense, Dropout # pyright: ignore[reportMissingImports]
from tensorflow.keras.models import Sequential# pyright: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split
import os

@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "spam.csv")

    df = pd.read_csv(csv_path, encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ['label','message']
    df['label'] = df['label'].map({'ham': 0, 'spam' : 1})
    return df


# Model Building 

def train_model(df):
    x = df['message'].values
    y = df['label'].values

    x_train,x_test,y_train,y_test = train_test_split(
        x,y , test_size= 0.2, random_state= 42
        )
    

    Vectorizer = TextVectorization(
        max_tokens = 5000,
        output_sequence_length = 100,
        standardize = 'lower_and_strip_punctuation'
    )
    

    Vectorizer.adapt(x_train)

    model = Sequential([
        Vectorizer,
        Embedding(5000,64),
        Dense(32,activation='relu'),
        Dropout(0.3),
        Dense(1,activation='sigmoid')
    ])


    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    loss, accuracy = model.evaluate(x_test,y_test,verbose = 0)

    return model,accuracy


## Interface 

st.title("📧 Spam Detection App (TensorFlow)")
st.write("Deep Learning based Spam Classifier")

df = load_data()
model, accuracy = train_model(df)

user_input = st.text_area("Enter Message")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message")
    elif len(user_input.split()) < 3:
        st.warning("Please enter a longer message")
    else:
        prob = model.predict([user_input])[0][0]

        if prob > 0.5:
            st.error(f"🚨 SPAM\nConfidence: {prob*100:.2f}%")
        else:
            st.success(f"✅ NOT SPAM\nConfidence: {(1-prob)*100:.2f}%")