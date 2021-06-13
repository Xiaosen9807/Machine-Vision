import cv2
import time
import mediapipe as mp
import handtrackingmodule as htm

Ptime = 0
Ctime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    ret, img = cap.read()
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[4])

    Ctime = time.time()
    fps = 1 / (Ctime - Ptime)
    Ptime = Ctime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
