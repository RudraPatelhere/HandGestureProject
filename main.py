import cv2
import mediapipe as mp
import pyttsx3
import threading
import time
import os

# Suppress TensorFlow Lite warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 200)  # Speech rate
engine.setProperty('volume', 1.0)  # Volume level

def speak(text):
    """Make the speaker announce the text."""
    engine.say(text)
    engine.runAndWait()

def speak_async(text):
    """Run the speak function in a separate thread to avoid blocking."""
    threading.Thread(target=speak, args=(text,)).start()

def is_finger_extended(tip, dip, mcp):
    """Determine if a finger is extended based on joint positions."""
    return tip.y < dip.y < mcp.y  # Finger is extended if TIP is above DIP and MCP

def classify_gesture(hand_landmarks, is_right_hand):
    """
    Classify gestures based on hand landmarks.
    """
    # Get key landmarks
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_MCP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP]
    index_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_dip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP]
    middle_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    ring_dip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_DIP]
    ring_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_MCP]
    pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]
    pinky_dip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_DIP]
    pinky_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP]

    # Determine which fingers are extended
    thumb_extended = thumb_tip.x < thumb_ip.x < thumb_mcp.x if is_right_hand else thumb_tip.x > thumb_ip.x > thumb_mcp.x
    index_extended = is_finger_extended(index_tip, index_dip, index_mcp)
    middle_extended = is_finger_extended(middle_tip, middle_dip, middle_mcp)
    ring_extended = is_finger_extended(ring_tip, ring_dip, ring_mcp)
    pinky_extended = is_finger_extended(pinky_tip, pinky_dip, pinky_mcp)

    # Gesture rules
    if pinky_extended and ring_extended and middle_extended and not index_extended and not thumb_extended:
        return "Hello"
    elif all([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
        return "Good Morning"
    elif index_extended and pinky_extended and not middle_extended and not ring_extended and not thumb_extended:
        return "I Love You"
    elif index_extended and pinky_extended and not middle_extended and not ring_extended:
        return "You Look Beautiful"
    elif not any([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
        return "So Sorry"
    elif middle_extended and not index_extended and not ring_extended and not pinky_extended and not thumb_extended:
        return "Fuck You"

    return "Unknown Gesture"

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    mp_drawing = mp.solutions.drawing_utils

    # Open the webcam feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam. Check your camera or permissions.")
        return

    print("Webcam successfully accessed. Press 'q' to exit.")
    last_gesture = None  # Track the last detected gesture
    last_spoken_time = 0  # Track the last time a gesture was announced

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame from the camera.")
            break

        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)

        # Resize the frame for faster processing
        reduced_frame = cv2.resize(frame, (640, 480))

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(reduced_frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand detection
        result = hands.process(rgb_frame)

        # Check for hands
        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = handedness.classification[0].label  # "Left" or "Right"
                is_right_hand = hand_label == "Right"

                # Classify the gesture
                gesture = classify_gesture(hand_landmarks, is_right_hand)

                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(reduced_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display the gesture on the frame
                cv2.putText(reduced_frame, f"Gesture: {gesture}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Speak the gesture every 3 seconds if it remains the same
                current_time = time.time()
                if gesture == last_gesture and current_time - last_spoken_time >= 3:
                    last_spoken_time = current_time
                    speak_async(gesture)
                elif gesture != last_gesture:
                    last_gesture = gesture
                    last_spoken_time = current_time
                    speak_async(gesture)

        # Display the frame
        cv2.imshow("Gesture Recognition with Timed Audio", reduced_frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed closed.")

if __name__ == "__main__":
    main()
