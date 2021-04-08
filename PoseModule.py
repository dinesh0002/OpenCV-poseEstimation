# Importing the required libraries
import cv2
import mediapipe as mp
import time

class poseDetector():

    def __init__(self, mode = False, upBody = False, smooth = True,
                 detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape  # height, width, channel
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

''' static_image_mode = False,
    upper_body_only = False,
    smooth_landmark = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
'''

def main():
    cap = cv2.VideoCapture('Resources/fitWalk.mp4')
    prev_frame_time = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[26]) # the arg in lmlist take the number from media pipe for different parts of the body recog
            cv2.circle(img, (lmList[26][1], lmList[26][2]), 15, (0, 0, 255), cv2.FILLED)
        '''this prints the specified number for tracking in red color '''

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break




if __name__ == "__main__":
    main()