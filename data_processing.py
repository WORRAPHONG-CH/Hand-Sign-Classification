# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 00:18:21 2023

@author: Worra
"""

import cv2 
import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

import pickle 
import mediapipe as mp 

mp_hands = mp.solutions.hands.Hands(static_image_mode=True,min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils


dir_data = "./dataset"

label_class = []
dataset_hand = []
img_list = []
count_notFound = 0

df = pd.DataFrame()

for dir_class in os.listdir(dir_data):
    dataset_path = os.path.join(dir_data,dir_class)
    print("Process path{}".format(dir_class))
    #print(dataset_path)
    for img_name in os.listdir(dataset_path):
        #print(dir_,img_name)
        img_path = os.path.join(dataset_path,img_name)
        #print(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        img_list.append(img)
        
        # MEDIA PIPE
        results = mp_hands.process(img)
        
        # if found hand 
        if results.multi_hand_landmarks:
            
            for hand_landmarks in results.multi_hand_landmarks:
                xy_landmark = []
                #xy_landmark = {}  # Dictionary to store the coordinates of each landmark
                
                for ind in range(len(hand_landmarks.landmark)): # 21 landmark
                    x_hand = hand_landmarks.landmark[ind].x
                    y_hand = hand_landmarks.landmark[ind].y
                    #print(ind)
                    # print("x{}".format(ind))
                    # print("y{}".format(ind))
                    # xy_landmark["x{}".format(ind)] = x_hand
                    # xy_landmark["y{}".format(ind)] = y_hand
                    
                    # xy_landmark.append(x_hand)
                    # xy_landmark.append(y_hand)
                    
                    xy_landmark.extend([x_hand,y_hand])
                    
                #df = df.append(xy_landmark,ignore_index=True)
                # label_class.append(dir_class)
                
                # label_class.append(dir_class)
                # df = df.append(pd.Series(xy_landmark),ignore_index=True)
                
            #dataset_hand.append(xy_landmark)
            
            label_class.append(dir_class)
            df = df.append(pd.Series(xy_landmark),ignore_index=True)
            
            
        
        else:
            count_notFound +=1
            print("NOT FOUND HAND",dir_class,":",img_name)

print("-------- Saving----------")

# dict_data = {"Landmark_Hand": dataset_hand,
#              "Label": label_class}

# df = pd.DataFrame(dict_data)

# Rename the columns with the desired format
col_name = []
for ind_col in range(len(hand_landmarks.landmark)):
    col_name.extend([f"X_{ind_col}",f"Y_{ind_col}"])

df.columns = col_name

# Join label col 
labels = pd.DataFrame({"Label": label_class})
df = df.join(labels)


df.to_csv("dataset_hand.csv",index=False)

print("-------- Finish ----------")
        
        
        
        