import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

wave_start_time = None
WAVE_DURATION = 3

def trigger_action():
    print("🚨 Wave detected for 3 seconds! Triggering action...")
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_detected = False

    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if hand_detected:
        if wave_start_time is None:
            wave_start_time = time.time()
        else:
            elapsed = time.time() - wave_start_time
            if elapsed >= WAVE_DURATION:
                trigger_action()
                wave_start_time = None
    else:
        wave_start_time = None

    cv2.putText(frame, "Wave for 3 seconds to trigger!", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Wave Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()