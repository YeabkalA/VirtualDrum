import numpy as np
import cv2
from filter import filter
import overlay_factory
import area_listener
import time
import image_processor
import ml_kit
import sound_player

# http://www.iamkarthi.com/opencv-experiment-virtual-on-screen-drums/

def read_transparent(file_name):
    src = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    print(src.shape)
   
    bgr = src[:,:,:3] # Channels 0..2
    alpha = src[:,:,3] # Channel 3
    result = np.dstack([bgr, alpha]) # Add the alpha channel

    return result

cap = cv2.VideoCapture(0)
ct = 0
listeners = []
img_process = image_processor.ImageProcessor()

'''
for offset in range(0, 100, 5):
    for i in range(200):
        listeners.append(area_listener.AreaListener((offset + i*10, i*10), 40))
'''

start_time = time.time()
base_saved = False

prev_time = start_time

shape_printed = False
al = area_listener.AreaListener((100,100), 40)
sp = sound_player.SoundPlayer()

while(True):
    compute_time_start = time.time()
    _, frame = cap.read()
    frame = img_process.horizontal_flip(frame)
    for i in range(100, 141):
        for j in range(100, 141):
            col = frame[i, j]
            if (not col[2] <= 240 and col[2] >= 215) or (not col[1] <= 170 and col[0] >= 130) or (not col[0] <= 255 and col[0] >= 245):
                frame[i, j] = np.asarray([255,255,255])
            else:
                frame[i, j] = np.asarray([0,0,0])

    if not base_saved and time.time() - start_time > 1:
        cv2.imwrite('passed.jpg', frame)
        al.set_base_image(frame)
        base_saved = True
    al.draw_area(frame)

    #if time.time() - prev_time > 0.1 and base_saved:
    if base_saved:
        prev_time = time.time()
        al.compare_difference_and_play_sound(frame)

    cv2.imshow('dst', frame)
    cv2.moveWindow('dst', 0,0)

    print(f'Compute time = {time.time() - compute_time_start}')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
