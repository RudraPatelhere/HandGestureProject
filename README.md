# Hand Gesture Recognition Project

This project is a real-time **hand gesture recognition system** that uses MediaPipe and OpenCV. 
It can detect specific hand gestures and provide audio feedback, making it interactive and useful for various applications.

## Screenshot
![Hand Gesture Recognition Screenshot](Hand1.png)

## How It Works

1. **Core Technologies**:
   - **MediaPipe**: Detects hand landmarks and provides highly accurate hand tracking.
   - **OpenCV**: Captures video input from your webcam and processes frames.
   - **Text-to-Speech (pyttsx3)**: Converts recognized gestures into spoken words.

2. **Recognized Gestures**:
   - **Hello**: Pinky, ring, and middle fingers extended.
   - **Good Morning**: All fingers extended (open palm).
   - **I Love You**: Index and pinky fingers extended.
   - **You Look Beautiful**: Both hands with index and pinky fingers extended.
   - **So Sorry**: All fingers curled into a fist.
   - **Middle Finger**: Middle finger extended (for fun or informal scenarios 😄).

3. **Output**:
   - Displays the recognized gesture on the screen.
   - Announces the gesture using audio feedback.
   - If the same gesture remains for more than 3 seconds, the system repeats the announcement.

---

## Setup and Usage

### Prerequisites
- Python 3.7 or later
- A webcam
- Libraries: `mediapipe`, `opencv-python`, `pyttsx3`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/RudraPatelhere/HandGestureProject.git
   cd HandGestureProject

