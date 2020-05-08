import cv2
from image_processor import ImageProcessor
def test_live():
    # if True:
    #     return
    cap = cv2.VideoCapture(0)

    while True:
        _, frame_orig = cap.read()
        frame_orig = ImageProcessor().horizontal_flip(frame_orig)
        
        cv2.imshow('Main', frame_orig)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

test_live()