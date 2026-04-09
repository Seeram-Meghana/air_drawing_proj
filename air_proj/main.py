import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Start camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not opening")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read camera")
        break

    # Flip image (mirror view)
    img = cv2.flip(img, 1)

    # Convert to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process hand detection
    results = hands.process(imgRGB)

    # Draw landmarks if detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img, handLms, mp_hands.HAND_CONNECTIONS
            )

    # Show output
    cv2.imshow("Hand Tracking", img)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release everything properly
cap.release()
cv2.destroyAllWindows()