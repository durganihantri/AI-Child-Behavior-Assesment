import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import torch
import librosa
import speech_recognition as sr
import noisereduce as nr
import pandas as pd
import plotly.express as px
from deepface import DeepFace
from pydub import AudioSegment
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Ensure Pydub uses ffmpeg
AudioSegment.converter = "/usr/bin/ffmpeg"

# Title & Instructions
st.title("🤗 AI Child Behavior Assessment")
st.markdown(
    """
    ### How to Use:
    1️⃣ Choose an **analysis type** below.  
    2️⃣ Upload the required file(s).  
    3️⃣ Click the **Analyze** button to process the data.  
    """
)

# Load AI Model for Speech Recognition
st.write("⏳ Loading AI Speech Model...")
try:
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    st.success("✅ AI Speech Model Loaded!")
except Exception as e:
    st.error(f"❌ Error loading speech model: {e}")

# ======================== DEFINE VIDEO ANALYSIS FUNCTION ========================
def analyze_video(video_path):
    """Processes video and extracts emotions with visualization"""
    st.write("🔎 Analyzing Emotions in Video...")
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    emotions_detected = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 10 == 0:  # Analyze every 10th frame
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotions_detected.append(analysis[0]['dominant_emotion'])
            except Exception as e:
                st.error(f"⚠️ DeepFace error: {e}")
        frame_count += 1

    cap.release()
    if emotions_detected:
        most_common_emotion = max(set(emotions_detected), key=emotions_detected.count)
        st.success(f"🧐 Most detected emotion: {most_common_emotion}")

        # Visualization
        emotion_counts = pd.Series(emotions_detected).value_counts()
        emotion_df = pd.DataFrame({'Emotion': emotion_counts.index, 'Count': emotion_counts.values})
        fig = px.bar(emotion_df, x='Emotion', y='Count', title="Emotion Distribution in Video", color='Emotion')
        st.plotly_chart(fig)
    else:
        st.warning("⚠️ No emotions detected. Try a different video.")

# ======================== DEFINE AUDIO ANALYSIS FUNCTION ========================
def transcribe_audio(audio_path):
    """Processes audio and extracts transcription with visualization"""
    try:
        st.write(f"🔎 Processing Audio File...")
        speech, sr = librosa.load(audio_path, sr=16000)

        # Enhanced Preprocessing
        speech = nr.reduce_noise(y=speech, sr=sr, prop_decrease=0.4)
        speech = librosa.effects.trim(speech)[0]
        speech = librosa.util.normalize(speech)

        st.write("🤖 Processing audio with AI model...")
        input_values = processor(speech, sampling_rate=16000, return_tensors="pt").input_values

        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        st.success(f"📝 Transcription (AI Model): {transcription}")

        # Visualization
        word_count = pd.Series(transcription.split()).value_counts()
        word_df = pd.DataFrame({'Word': word_count.index, 'Count': word_count.values})
        fig = px.bar(word_df, x='Word', y='Count', title="Word Frequency in Transcription", color='Word')
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"⚠️ Error in AI Speech Processing: {e}")

# ======================== USER SELECTS ANALYSIS MODE ========================
analysis_option = st.radio(
    "Select Analysis Type:",
    ["📹 Video Only (Facial Emotion)", "🎤 Audio Only (Speech Analysis)", "🎬 Video & Audio (Multimodal)"]
)

# ======================== VIDEO ONLY ANALYSIS ========================
if analysis_option == "📹 Video Only (Facial Emotion)":
    st.header("📂 Upload a Video for Emotion Analysis")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_file.read())
            video_path = temp_video.name
        st.success("📂 Video uploaded successfully!")

        if st.button("Analyze Video"):
            analyze_video(video_path)

# ======================== AUDIO ONLY ANALYSIS ========================
elif analysis_option == "🎤 Audio Only (Speech Analysis)":
    st.header("🎤 Upload an Audio File for Speech Analysis")
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            audio_path = temp_audio.name
        st.success("🎤 Audio uploaded successfully!")

        if st.button("Analyze Audio"):
            transcribe_audio(audio_path)

# ======================== MULTIMODAL ANALYSIS (VIDEO + AUDIO) ========================
elif analysis_option == "🎬 Video & Audio (Multimodal)":
    st.header("🎥 Upload a **Single File** for Video & Audio Combined Analysis")
    multimodal_file = st.file_uploader("Upload a **video file with audio**", type=["mp4", "avi", "mov"])

    if multimodal_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(multimodal_file.read())
            multimodal_path = temp_file.name

        st.success("✅ Multimodal file uploaded successfully!")

        if st.button("Analyze Video & Audio Together"):
            def analyze_multimodal(multimodal_path):
                st.write("🔎 Extracting Video & Audio...")

                # Extract Video Emotion
                video_emotions = analyze_video(multimodal_path)

                # Extract Audio for Speech Processing
                audio_transcription = transcribe_audio(multimodal_path)

                # Multimodal Analysis Visualization
                st.header("🔍 Multimodal Analysis Results")
                if not video_emotions or not audio_transcription:
                    st.error("❌ Could not extract both Video & Audio insights.")
                    return

                # Emotion-Speech Comparison
                speech_emotion = "Neutral"
                if any(word in audio_transcription.lower() for word in ["angry", "mad"]):
                    speech_emotion = "Angry"
                elif any(word in audio_transcription.lower() for word in ["happy", "excited"]):
                    speech_emotion = "Happy"
                elif any(word in audio_transcription.lower() for word in ["sad", "crying"]):
                    speech_emotion = "Sad"

                fig = px.pie(
                    names=["Video Emotion", "Speech Emotion"],
                    values=[len(video_emotions), 1],
                    title=f"Comparison: Video ({video_emotions[0]}) vs. Speech ({speech_emotion})"
                )
                st.plotly_chart(fig)

            analyze_multimodal(multimodal_path)
