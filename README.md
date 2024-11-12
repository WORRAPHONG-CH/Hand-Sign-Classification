# Hand-Sign-Classification
- This project aims to applied Mediapipe with machine learning models and image processing to classify the hand sign into 5 classes (“LIKE”, “DISLIKE”, “OK”, “LOVE”, “STOP”).
- Identified the best model by comparing **Random Forest**,**SVC (Support Vector Classification)**, and **Naive Bayes**.
- ![p1](https://github.com/user-attachments/assets/8539f5dc-37c9-41eb-85d2-46115188f1aa)

## Concept
-   Extract frame and applied image processing techniques to preprocess the images.
-   The hand signs including “LIKE”, “DISLIKE”, “OK”, “LOVE”, “STOP”, with 300 images of each hand sign collected for use in training the model.
-   Using MediaPipe library to detect and store XY position of 21 hand landmarks in csv file.
-   RandomForest model achieved an accuracy score of 0.99
-   Test hand signs on camera with the best model.

![overview](https://github.com/user-attachments/assets/d4a81505-4a8f-453b-a1fe-1bdcbf956c04)

## Results
![class0](https://github.com/user-attachments/assets/76df6e79-9de4-4f8e-b2d5-8943e7c66746)
![class1](https://github.com/user-attachments/assets/b7a94a83-6ba3-414b-aa3f-e47446301f14)

![class2](https://github.com/user-attachments/assets/a29bad11-fd16-416f-b2cb-5fd847f09c14)
![class3](https://github.com/user-attachments/assets/9aed85da-8c8a-436e-bc54-25e72bc8ba66)
![class4](https://github.com/user-attachments/assets/f6bb6480-a0a7-4ca3-a949-d1c063898b1a)
