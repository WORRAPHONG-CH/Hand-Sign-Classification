import cv2 
import numpy as np 
import pandas as pd 
import mediapipe as mp

hands = mp.solutions.hands.Hands(max_num_hands=2,min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cv2.namedWindow("Frame",cv2.WINDOW_FREERATIO)

hand_count = 0

while(cap.isOpened()):
    ret,frame = cap.read()
    frame = cv2.flip(frame,1)
    
    if ret == True:
        h,w,ch = frame.shape
        frame_display = frame.copy()
        #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = hands.process(frame)
        hand_detect = results.multi_hand_landmarks
        
        # Check if any hands are detected
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            cv2.putText(frame_display,"Hand Count:{}".format(hand_count),(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            # Iterate over the detected hands
            for hand_landmark,handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                # Determine the handedness(left or right hand)
                hand_label = handedness.classification[0].label
                
                # hand_landmark.landmark   x,y,z-> 21 set [scale normalize 0-1]
                #print("--------------------")
                
                # Get x,y pixel of hand 
                landmark_list = []
                for ind,landmark in enumerate(hand_landmark.landmark,0):
                    x_hand = int(landmark.x * w) 
                    y_hand = int(landmark.y * h)
                    #x_hand = min(int(landmark.x * w),w-1) # -1 for ensure that not more than size image
                    #y_hand = min(int(landmark.y * h),h-1)
                    # min ->  compares the calculated xy coordinate of the landmark with the maximum allowable coordinate and returns the smaller value. 
                    # This step ensures that the calculated x-coordinate does not exceed the image width, preventing any out-of-bounds error
                    #landmark_list.append(np.array((x_hand,y_hand),dtype="uint8"))
                    landmark_list.append((x_hand,y_hand))
                    #print(ind,":",landmark)
                    #print(ind,":",y_hand)
                    
                #print("-----------")
                
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
                
                # Get index finger landmark
                index_finger_landmark = hand_landmark.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                x_index_finger = index_finger_landmark.x
                y_index_finger = index_finger_landmark.y
                
                x_pixel = int(x_index_finger * w)
                y_pixel = int(y_index_finger * h)
                
                # Draw hand landmarks on the frame
                #print("x:{} | y:{}".format(x_pixel,y_pixel))
                cv2.putText(frame_display,"Index Finger",(x_pixel,y_pixel),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,0,0),1)
                mp_draw.draw_landmarks(frame_display, hand_landmark,
                                    mp.solutions.hands.HAND_CONNECTIONS)
                
                label_hand = "LEFT HAND" if x_box < frame.shape[1] /2 else "RIGHT HAND"
                
                cv2.putText(frame_display,hand_label,(x_box-25,y_box-25),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)
                '''for h_landmark in hand_landmark:
                    print(hand_landmark.x,hand_landmark.y)
                    print("---------------")'''
                not_empty_mask = np.any(mask)
                not_empty_roi = np.any(roi_hand)
                
                if not_empty_mask == True and not_empty_roi == True:
                    cv2.imshow("Mask Hand",mask)
                    cv2.imshow("Hand ROI",roi_hand) # need to show in loop because if it's not detect hand there will be no roi_hand 
                
        cv2.imshow("Frame",frame_display)
        
        
        
        button = cv2.waitKey(1)
        if button & 0xFF == 27:
            break
    
    else:
        print("frame not return")
        
cap.release()
cv2.destroyAllWindows()