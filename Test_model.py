# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 14:17:14 2023

@author: Worra
"""

import cv2 
import numpy as np 
import pandas as pd
import mediapipe as mp
import pickle
import joblib

mp_hands= mp.solutions.hands.Hands(static_image_mode = False,
                                   max_num_hands=2,
                                   min_detection_confidence=0.5)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# load model 
feature_name = ["X_0", "Y_0", "X_1", "Y_1", "X_2", "Y_2", "X_3", "Y_3", "X_4", "Y_4",
       "X_5", "Y_5", "X_6", "Y_6", "X_7", "Y_7", "X_8", "Y_8", "X_9", "Y_9",
       "X_10", "Y_10", "X_11", "Y_11", "X_12", "Y_12", "X_13", "Y_13", "X_14", "Y_14", 
       "X_15", "Y_15", "X_16", "Y_16", "X_17", "Y_17", "X_18", "Y_18",
       "X_19", "Y_19", "X_20", "Y_20"]

#feature_name = [str(name) for name in feature_name]

#df_test = pd.DataFrame(columns=feature_name)
#df_test = pd.DataFrame()
model_forest = joblib.load("forest_model.joblib")

# with open('./model_forest.pkl', 'rb') as file:
#     model_forest = pickle.load(file)
    
#model_forest = pickle.load(open("/model_forest.pkl","rb"))

label_class = ["LIKE","DISLIKE","OK","LOVE","STOP"]
hand_count=0
count_row = 0

while(cap.isOpened()):
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    
    if ret == True:
        h,w,ch = frame.shape
        frame_display = frame.copy()
        #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = mp_hands.process(frame)
        hand_detect = results.multi_hand_landmarks
        
        # Check if any hands are detected
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            cv2.putText(frame_display,"Hand Count:{}".format(hand_count),(50,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            # Iterate over the detected hands
            for hand_landmark,handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                # Determine the handedness(left or right hand)
                hand_label = handedness.classification[0].label
                
                # hand_landmark.landmark   x,y,z-> 21 set [scale normalize 0-1]
                #print("--------------------")
                
                # Get x,y pixel of hand 
                test_landmark = []
                landmark_list = []
                for ind,landmark in enumerate(hand_landmark.landmark,0):
                    x_landmark = landmark.x
                    y_landmark = landmark.y
                    x_hand = int(landmark.x * w) 
                    
                    y_hand = int(landmark.y * h)
                    
                    landmark_list.append((x_hand,y_hand))
                    
                    test_landmark.extend([x_landmark,y_landmark])
                    
                count_row += 1
                # Predict Class
                df_test = pd.DataFrame()
                df_test= df_test.append(pd.Series(test_landmark),ignore_index= True)
                reshape_test_landmark = np.array(test_landmark).reshape(1,-1)
                #reshape_test_landmark = np.tile(test_landmark,(1,21))
                prediction = model_forest.predict(df_test)
                #prediction = prediction.reshape(-1)
                prediction = prediction.item()
                label_predict = label_class[prediction]
                #print(label_predict)
                
                cv2.putText(frame_display,"SIGN: {}".format(label_predict),(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
                # Calculate bounding rectangle using cv2.boundingRect()
                x_box,y_box,w_box,h_box = cv2.boundingRect(np.array(landmark_list))
                
                # Create mask of ROI 
                mask = np.zeros_like(frame)
                
                y1_roi, y2_roi = y_box-20, y_box + h_box + 20
                x1_roi, x2_roi = x_box-20, x_box + w_box +20
                roi_hand = frame[y1_roi:y2_roi,x1_roi:x2_roi] #[y1:y2,x1:x2]
                mask[y1_roi:y2_roi,x1_roi:x2_roi] = frame[y1_roi:y2_roi,x1_roi:x2_roi]
                
                # Draw bounding box around the hand
                cv2.rectangle(frame_display,(x1_roi,y1_roi),(x2_roi,y2_roi),(0,255,255),2)
                
                
                # Draw hand landmarks on the frame
                #print("x:{} | y:{}".format(x_pixel,y_pixel))
                
                mp_draw.draw_landmarks(frame_display, hand_landmark,
                                    mp.solutions.hands.HAND_CONNECTIONS)
                
                
                cv2.putText(frame_display,hand_label,(x_box-25,y_box-25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)
                '''for h_landmark in hand_landmark:
                    print(hand_landmark.x,hand_landmark.y)
                    print("---------------")'''
                
                
               
                
        cv2.imshow("Frame",frame_display)
        
        
        
        button = cv2.waitKey(1)
        if button & 0xFF == 27:
            break
    
    else:
        print("frame not return")
        
cap.release()
cv2.destroyAllWindows()