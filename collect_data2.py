import cv2 
import os
import mediapipe as mp 
import numpy as np

# Media pipe 
mp_hands = mp.solutions.hands.Hands(max_num_hands=2,min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Camera 
cap = cv2.VideoCapture(0)

# ================== Function ================== #
def real_pos_hand(hand_landmark,w,h):
    # hand_landmark.landmark   x,y,z-> 21 set [scale normalize 0-1]
    
    landmark_list = []
    for ind,landmark in enumerate(hand_landmark.landmark,0):
        x_hand = int(landmark.x * w)
        y_hand = int(landmark.y * h)
        landmark_list.append((x_hand,y_hand))
        
    return np.array(landmark_list)

# SET PATH For collect dataset 
dir_dataset = "./dataset"

if not os.path.exists(dir_dataset):
    os.makedirs(dir_dataset)
    print("=========== FINISH CREATE PATH ===========")

else: 
    print("=========== PATH ALREADY EXISTS ===========")
    
label_class = ["LIKE","DISLIKE","OK","LOVE","STOP"]
num_class = len(label_class)
dataset_size = 200
time_save = dataset_size * 10 

count_class = 0
for j in range(num_class):
    dir_class = os.path.join(dir_dataset,str(j))
    if not os.path.exists(dir_class):
        os.makedirs(dir_class)
        print("=========== FINISH CREATE PATH CLASS ===========")
          
    else:
        print("=========== PATH CLASS ALREADY EXISTS [{}] ===========".format(label_class[j]))
        
    done = False
    while(True):
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        if ret == True:
            h,w,ch = frame.shape
            frame_display = frame.copy()
            
            # create result mediapipe 
            results = mp_hands.process(frame) # Process image 
            hand_detect = results.multi_hand_landmarks # Check if any hands are detected
            
            # Check if any hands are detected 
            if results.multi_hand_landmarks:
                hand_count = len(results.multi_hand_landmarks)
                cv2.putText(frame_display,"HAND COUNT : {}".format(hand_count),(50,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,100),2)
                
                # Iterate over the detected hands
                for hand_landmark,handdedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                    
                    # Determine the handdedness(left or right hands)
                    hand_label = handdedness.classification[0].label
                   
                    # hand_landmark.landmark   x,y,z-> 21 set [scale normalize 0-1]
                    
                    # get real pos 
                    landmark_list = real_pos_hand(hand_landmark, w, h)
                    #print(landmark_list)
                    
                    # Calculate bounding rectangle using cv2.boundingRect()
                    x_box,y_box,w_box,h_box = cv2.boundingRect(landmark_list)
                    
                    x1_roi,x2_roi = x_box -20 ,x_box + w_box +20 
                    y1_roi,y2_roi = y_box -20 ,y_box + h_box +20
                    
                    # Create ROI 
                    roi_hand = frame[y1_roi:y2_roi,x1_roi:x2_roi] #[y1:y2,x1:x2]
                    
                    # Drawing 
                    mp_draw.draw_landmarks(frame_display, hand_landmark,
                                           mp.solutions.hands.HAND_CONNECTIONS)
                    cv2.rectangle(frame_display,(x1_roi,y1_roi),(x2_roi,y2_roi),(0,255,255),2)
                    
                    
                    
                    not_empty_roi = np.any(roi_hand)
                    
                    if not_empty_roi == True:
                        cv2.imshow("ROI HAND",roi_hand)
            
            
            cv2.putText(frame_display,"Collect Dataset Press S ",(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            cv2.imshow("RESULT",frame_display)
            
            
            if cv2.waitKey(1) & 0xFF == 27:
                
                key_save = 0
                break
            
            if cv2.waitKey(1) & 0xFF == ord("s"):
                key_save = 1
                break
            
        else:
            pass
              
            

cap.release()
cv2.destroyAllWindows()