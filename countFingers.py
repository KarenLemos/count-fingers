import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)

mpDrawing = mp.solutions.drawing_utils
mpHands = mp.solutions.hands

fingerTip = [4,8,12,16,20]

hands = mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

def countFingers(image, handLandmarks, handNo=0):
    if handLandmarks:
        landmarks = handLandmarks[handNo].landmark
        # print(landmarks)

        fingers = []

        for index in fingerTip:
            fingerTipY = landmarks[index].y
            fingerBottomY = landmarks[index - 2].y

            if index != 4:
                if fingerTipY < fingerBottomY:
                    fingers.append(1)
                    print("Dedo com id ", index, "está aberto")

                if fingerTipY > fingerBottomY:
                    fingers.append(0)
                    print("Dedo com id ", index, "está fechado")

        totalFingers = fingers.count(1)

        text = f'Dedos: {totalFingers}'
        cv2.putText(image, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

def drawHandLanmarks(image, handLandmarks):
    if handLandmarks:
        for landmarks in handLandmarks:
            mpDrawing.draw_landmarks(image, landmarks, mpHands.HAND_CONNECTIONS)

while True:
    success, image = webcam.read()

    image = cv2.flip(image, 1)

    results = hands.process(image)

    handLandmarks = results.multi_hand_landmarks
    drawHandLanmarks(image, handLandmarks)

    countFingers(image, handLandmarks)

    cv2.imshow('Webcam', image)

    key = cv2.waitKey(1)

    if key == 32:
        break

cv2.destroyAllWindows()