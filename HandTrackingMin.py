# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 12:31:50 2021

@author: Aditya
"""
import cv2
import mediapipe as mp
# import time
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


with mpHands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        print(results.multi_hand_landmarks)
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)    
        # cTime = time.time()
        # fps = 1/(cTime-pTime)
        # pTime = cTime
        
        # cv2.putText(img,str(int(fps)))
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == ord('q'):
            # cv2.destroyAllWindows()
            break
        
cap.release()
cv2.destroyAllWindows()



