import cv2
from image_processor import ImageProcessor

SAMPLES_TO_COLLECT = 30

def collect_samples():
    cap = cv2.VideoCapture(0)
    
    img_process = ImageProcessor()
    count = -1

    frames = []
    while True:
        print(f"Count = {count}")
        
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = img_process.horizontal_flip(frame)

        cv2.imshow('Frame', frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            count = 1
        
        if count > 0 and count <= SAMPLES_TO_COLLECT:
            frames.append(frame)
            count += 1
        elif count > SAMPLES_TO_COLLECT:
            break

        if key & 0xFF == ord('q'):
            break
    
    ind = 0
    while True:
        cv2.imshow('Part', frames[ind])
        key = cv2.waitKey(1)
        if key == ord('s'):
            ind += 1

collect_samples()