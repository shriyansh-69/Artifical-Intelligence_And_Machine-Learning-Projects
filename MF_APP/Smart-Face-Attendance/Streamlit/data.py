import streamlit as st
import pandas as pd

st.title("Data analyzer")

file = st.file_uploader("Upload Your CSV File", type = ['CSV'])

if file:
    df = pd.read_csv(file)
    st.subheader("Data Preview")
    st.dataframe(df)

if file:
    st.subheader("Summary Stat")
    st.write(df.describe())

if file:
    s = df["ethnicity"].unique()
    Selected = st.selectbox("Fliter By ethnicity", s)
    filtered = df[df['ethnicity'] == Selected ]
    st.dataframe(filtered)

if file:
    val = df["status"].unique()
    valu = st.selectbox("Fliter By Status", val)
    filter = df[df['status'] == valu]
    st.dataframe(filter)
