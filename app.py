from flask import Flask, request, jsonify
import cv2
import speech_recognition as sr
from deepface import DeepFace

app = Flask(__name__)

@app.route('/analyze_face', methods=['POST'])
def analyze_face():
    file = request.files['video']
    video_path = "uploaded_video.mp4"
    file.save(video_path)
    
    cap = cv2.VideoCapture(video_path)
    emotions = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        analysis = DeepFace.analyze(frame, actions=['emotion'])
        emotions.append(analysis[0]['dominant_emotion'])
    
    cap.release()
    return jsonify({"emotions": emotions})

@app.route('/analyze_speech', methods=['POST'])
def analyze_speech():
    file = request.files['audio']
    audio_path = "uploaded_audio.wav"
    file.save(audio_path)

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
    
    return jsonify({"transcribed_text": text})

if __name__ == '__main__':
    app.run(debug=True)
