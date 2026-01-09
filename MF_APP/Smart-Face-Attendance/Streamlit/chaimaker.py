import streamlit as st

st.title("Yo")

if st.button("Make Chai"):
    st.success("Your chai is Brewd")

add = st.selectbox("Do you want to add masala?", ['Yes', 'No'])


if add == 'Yes' :
    st.write("Masala Is Add To Your Chai ")
else:
    st.write("Masala Is Not  There You Son Of Gun")

tea_type = st.radio("Pick Your Base Nigga:- ",['Milk','Doodh','Japanase_Milk'])    

st.write(f"Selected {tea_type}")

flavour = st.selectbox("Choose Your Favourite Flavour",["Adrak","Caramel","Vanila"])

st.write(f"Selected Flavour is {flavour}")

sugar = st.slider("Pick Your Sugar Level",0,5,3)
st.write(f"Selected Sugar Level {sugar}")

Cup = st.number_input("How Many Cup's", min_value= 1,max_value=10, step = 1)
st.write(f"How Many {Cup}")

Name = st.text_input("Enter Your Name:- ")
if Name:
    st.write(f"Welcome,{Name} ! Your Chai Is On The Way ")

DOB = st.date_input("Select Your Date Of Birth")
st.write(f"Your Date-Of-Birth {DOB}")


