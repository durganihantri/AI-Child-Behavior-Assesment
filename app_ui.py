import streamlit as st
import requests

st.title("AI-Powered Child Behavior Assessment")

uploaded_video = st.file_uploader("Upload a child's video", type=["mp4"])
uploaded_audio = st.file_uploader("Upload a child's audio", type=["wav"])

if uploaded_video:
    st.video(uploaded_video)
    if st.button("Analyze Video"):
        files = {"video": uploaded_video}
        response = requests.post("http://127.0.0.1:5000/analyze_face", files=files)
        st.json(response.json())

if uploaded_audio:
    st.audio(uploaded_audio)
    if st.button("Analyze Audio"):
        files = {"audio": uploaded_audio}
        response = requests.post("http://127.0.0.1:5000/analyze_speech", files=files)
        st.json(response.json())
