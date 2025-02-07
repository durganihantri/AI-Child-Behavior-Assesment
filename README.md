# AI-Child-Behavior-Assessment

This project is a web-based AI tool designed to analyze children's facial expressions and language development for early mental health assessment.

## Features
- AI-powered **Facial Emotion Detection** using DeepFace
- **Speech-to-Text Analysis** for child language development
- Flask API for backend processing
- Streamlit-based web interface

## Setup Instructions
### 1. Install Required Libraries
```bash
pip install flask streamlit tensorflow torch torchvision opencv-python numpy pandas speechrecognition deepface transformers
```

### 2. Run the Backend (Flask API)
```bash
cd backend
python app.py
```

### 3. Run the Frontend (Streamlit App)
```bash
cd frontend
streamlit run app_ui.py
```

## Deployment Guide
- Backend can be deployed on **Hugging Face Spaces**.
- Frontend can be hosted on **Streamlit Cloud** or **Netlify**.

## License
MIT License
