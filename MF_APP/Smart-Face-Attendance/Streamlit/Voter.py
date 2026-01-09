import streamlit as st

st.title("Chai Taste Poll")

col1 , col2 = st.columns(2)

with col1:
    st.header("Masala Chai")
    st.image("https://imgs.search.brave.com/6wzD6W"
    "-f4JFJmoRA62lRFObwlwCN05Lv0Mzb3ticg0o/rs:fit:500:0:1:0"
    "/g:ce/aHR0cHM6Ly93YWxs/cGFwZXJhY2Nlc3Mu/Y29tL2Z1bGwvMTIy/NTU1LmpwZw"
    ,width=200)
    vote1 = st.button("Vote Masala Chai")

with col2:
    st.header("Adrak Chai")
    st.image("https://imgs.search.brave.com/yvQiwvm1R7TT8oHchRpN2V6cy8tQi_AbGRoqFjdnki8/rs:fit:500:0:1:0/g:ce/" \
    "aHR0cHM6Ly9wcmV2/aWV3LnJlZGQuaXQv/c2hvdy1tZS15b3Vy/LWFic29sdXRlLWZh/dm9yaXRlLWJsZWFj/aC1waWN0dXJlLXlv/dS1oYXZlLW9yLXY" \
    "w/LWNuYWtmeGQ2c2p0/ZTEuanBlZz9mb3Jt/YXQ9cGpwZyZhdXRv/" \
    "PXdlYnAmcz00ZjA3/ODg2ZjBlOTMxZTk0/ZmEyZGM5MWQzMjJh/NmJhMDAxNjU3MWMz",width=200)
    vote2 = st.button("Vote Adrak chai")

if vote1:
    st.success("Thanks For Voting Masala Chai")
elif vote2:
    st.success("Thanks For Voting Adrak Chai")


name = st.sidebar.text_input("Enter Your Name:- ")
tea = st.sidebar.selectbox("What TEA Do You Want", ["Adrak","Masala","Choco"])

st.success(f"Welcome {name}! And Your {tea} Is Getting Ready")

with st.expander("Show Chai Making  Instruction"):
    st.write(""" 
    1. Boil Water With Tea Leave's
    2. Add Milk And Species
    3. Serve Hot
""")
    

st.markdown('### Welcome To Chai App')
st.markdown('> BlockQuote')