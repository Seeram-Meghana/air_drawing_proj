import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 720)

xp, yp = 0, 0
canvas = None
draw_color = (255, 0, 255)

smoothening = 1  # VERY IMPORTANT (no lag)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    # -------- TOP BAR --------
    cv2.rectangle(img, (0, 0), (960, 80), (30, 30, 30), -1)

    colors = [
        (255, 0, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 255, 255),
        (0, 0, 255)
    ]

    for i, color in enumerate(colors):
        cv2.circle(img, (80 + i*120, 40), 30, color, -1)

    # CLEAR
    cv2.rectangle(img, (750, 15), (930, 65), (0, 0, 0), -1)
    cv2.putText(img, "CLEAR", (770, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # -------------------------

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            h, w, c = img.shape
            lm_list = []

            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            x1, y1 = lm_list[8]
            x2, y2 = lm_list[12]

            # Finger states
            index_up = lm_list[8][1] < lm_list[6][1]
            middle_up = lm_list[12][1] < lm_list[10][1]

            # 🎯 DRAW POINTER (VERY IMPORTANT)
            cv2.circle(img, (x1, y1), 12, (0, 255, 255), cv2.FILLED)

            # ✌️ SELECTION MODE
            if index_up and middle_up:
                xp, yp = 0, 0

                if y1 < 80:
                    for i in range(len(colors)):
                        if (50 + i*120) < x1 < (110 + i*120):
                            draw_color = colors[i]

                    if 750 < x1 < 930:
                        canvas[:] = 0

            # ☝️ DRAW MODE
            elif index_up and not middle_up:

                if y1 < 80:
                    xp, yp = 0, 0
                    continue

                # NO DELAY SMOOTHING
                x1 = int(xp + (x1 - xp) / smoothening)
                y1 = int(yp + (y1 - yp) / smoothening)

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                # STRONG COLOR LINE
                cv2.line(canvas, (xp, yp), (x1, y1), draw_color, 10)

                xp, yp = x1, y1

            else:
                xp, yp = 0, 0

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    else:
        xp, yp = 0, 0

    # STRONG MERGE (no fading)
    img = cv2.add(img, canvas)

    cv2.imshow("Air Drawing FINAL FIXED", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        cv2.imwrite("air_drawing.png", canvas)
        print("Saved!")

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()