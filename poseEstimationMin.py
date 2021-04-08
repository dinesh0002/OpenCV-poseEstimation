# Importing the required libraries
import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# setting the capture video setting and adjusting
cap = cv2.VideoCapture('Resources/fitDan3.mp4')  # getting the video from the resources file from the local computer
#  cap.set(3, 640)  # initializing the width
#  cap.set(4, 480)  # initializing the height
''' videos are already rendered into 640x480 px from online'''

# used to record the time when we processed the last frame
prev_frame_time = 0
# used to record the time when we processed current frame
new_frame_time = 0

# since it is a video looping it with while loop as True
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # mediapipe is in RGB so convert to BGR 2 RGB
    results = pose.process(imgRGB)

    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):  # id, landmark(lm), enumerate is for the loop count
            h, w, c = img.shape  # height, width, channel
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    # code for the FPS display
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):  # as the wait key arg is increased then fps is decreased in the video
        break
