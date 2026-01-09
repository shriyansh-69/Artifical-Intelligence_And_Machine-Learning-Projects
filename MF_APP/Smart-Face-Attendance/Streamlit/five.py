import streamlit as st  
import requests

st.header("Live Currency Converter")
amount = st.number_input("Enter The Amount In INR", min_value=1)

target_currency = st.selectbox("Convert To:", ["USD","EUR","GBP","JPY"])


if st.button("Convert"):
    url =  "https://api.exchangerate-api.com/v4/latest/INR"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        rate = data["rates"][target_currency]
        converted = rate * amount
        st.write(f"{amount} INR = {converted}  {target_currency}")
    else:
        st.error("Failed To Fetch Data")


