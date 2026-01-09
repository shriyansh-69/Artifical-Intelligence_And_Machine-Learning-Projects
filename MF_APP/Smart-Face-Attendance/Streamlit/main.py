import streamlit as st

st.title("Hey Buddy, Are You Crazy Fun?")
st.subheader("Brewed With Streamlit")
st.text("What's up, Friend?")
st.write("Pick whatever color you like!")

# Selectbox with friendly options
color_choice = st.selectbox("Choose Your Favorite Color", ["Black", "Red", "India"])
st.write(f"You picked: {color_choice}")

st.success("Great Choice!")

