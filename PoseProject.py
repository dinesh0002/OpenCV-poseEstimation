import cv2
import time
import PoseModule as pm

''' using the py file PoseModule as an module for the upcoming projects as well as we are importing the 
module as import PoseModule as pm for initializing the poseDetector to run the test code what we have written 
in the PoseModule '''

cap = cv2.VideoCapture('Resources/running.mp4')
prev_frame_time = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[14])
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break