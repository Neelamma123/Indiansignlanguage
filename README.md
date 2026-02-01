Indian Sign Language Recognition System
📖 Introduction
This project presents a computer vision and deep learning–based Indian Sign Language (ISL) Recognition System that detects hand gestures using a webcam and recognizes alphabets (A–Z) and digits (1–9).
The system also converts recognized signs into voice output, enabling better communication between hearing-impaired users and non–sign-language users.

❓ Problem Statement
Communication between hearing-impaired individuals and the general public is often challenging due to limited understanding of sign language.
This project aims to bridge this gap by developing a real-time system that recognizes Indian Sign Language gestures and converts them into readable text and speech.

🎯 Objectives
To recognize Indian Sign Language alphabets (A–Z) and digits (1–9).
To detect both one-hand and two-hand gestures using MediaPipe and CVZone.
To apply deep learning techniques for accurate gesture classification.
To generate voice output for the recognized signs.
To provide a user-friendly interface using Streamlit.
🛠️ Technologies Used
Python
OpenCV
MediaPipe
CVZone
TensorFlow / Keras
NumPy
Streamlit
gTTS / pyttsx3
📂 Project Structure
HandSignDetection/ │ ├── app.py ├── datacollection/ ├── Model/ ├── Data/ ├── test.py ├── requirements.txt └── README.md
▶️ How to Run the Project
Step 1: Install Required Libraries
bash pip install -r requirements.txt

Step 2: Run the application
bash streamlit run app.py

##📊 Output

Real-time hand gesture detection using webcam.
Recognition of ISL alphabets and digits.
Display of recognized sign on the screen.
Audio output of the detected gesture.
##⚠️ Limitations

Requires proper lighting conditions.
Accuracy may decrease with complex backgrounds.
Limited to predefined alphabets and digits.
##🚀 Future Enhancements . Recognition of full words and sentences. . Support for dynamic gestures. . Mobile application implementation. . Improved accuracy using advanced deep learning models.
