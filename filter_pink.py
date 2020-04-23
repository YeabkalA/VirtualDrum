import cv2 
import numpy as np 
import time
import image_processor
  
cap = cv2.VideoCapture(0) 
  
img_process = image_processor.ImageProcessor()

prev_time = 0
while(1): 
    _, frame = cap.read() 
    frame = img_process.horizontal_flip(frame)
    # It converts the BGR color space of image to HSV color space 
    col = None
    for i in range(80, 130):
        for j in range(80, 130):
            if i == 100 and j==100:
                if time.time() - prev_time > 3:
                    prev_time = time.time()
                    print(frame[i, j])
            col = frame[i, j]
        
            if (not col[2] <= 240 and col[2] >= 215) or (not col[1] <= 170 and col[0] >= 130) or (not col[0] <= 255 and col[0] >= 245):
                frame[i, j] = np.asarray([255,255,255])
            else:
                frame[i, j] = np.asarray([0,0,0])
    
    #cv2.imshow('frame', frame) 
    cv2.imshow('mask', frame) 
    #cv2.imshow('result', result) 
      
    key = cv2.waitKey(1)
    if key == ord('s'):
        print(col)
  
cv2.destroyAllWindows() 
cap.release() 