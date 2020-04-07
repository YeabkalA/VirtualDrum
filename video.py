import numpy as np
import cv2
from filter import filter
import overlay_factory

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
while(True):
    _, frame = cap.read()
    alpha = 0.1
    
    hihat = read_transparent('images/hihat.png')
    hihat = cv2.resize(hihat, (hihat.shape[1]//3, hihat.shape[0]//3))
    snare = read_transparent('images/snare3.png')
    #snare = cv2.resize(snare, (snare.shape[1]//2, snare.shape[0]//2))
    snare2 = read_transparent('images/snare2.png')
    snare2 = cv2.resize(snare2, (snare2.shape[1]//2, snare2.shape[0]//2))
    #drum = cv2.resize(drum, (drum.shape[1]//2, drum.shape[0]//2))
    #drum = cv2.cvtColor(drum, cv2.COLOR_BGR2HSV )

    '''
    factory = overlay_factory.OverlayFactory()
    factory.set_base_image(frame)
    factory.add_overlay(hihat, (600, 400))
    factory.add_overlay(snare, (0, 400))
    factory.add_overlay(snare2, (300, 400))

    combined = factory.produce_overlay()
    '''

    cv2.imshow('dst', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
