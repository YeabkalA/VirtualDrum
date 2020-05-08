from area_listener import DrumArea
from area_listener import AreaListener
import cv2
import numpy as np
from tensor_sample_collector import get_file_name
from tensor_sample_collector import STICK_DATATYPE, NO_STICK_DATATYPE, BLURRY_DATATYPE

def get_pixels_list(img):
    pixels = []
    for i in range(0, len(img)):
        for j in range(0, len(img)):
            pixels.append(img[i][j])
    
    return pixels

def hist(img, bin_width = 1):
    img = cv2.resize(img, (32,32))
    pixels = get_pixels_list(img)
    bin_conts = [0 for i in range(256//bin_width)]
    for pixel in pixels:
        pixel_bin = pixel // bin_width
        bin_conts[pixel_bin] += 1
    
    lens =[]
    for ind, cont in enumerate(bin_conts):
        lens.append(cont//1)
        #print(f'{str(ind).rjust(4)}\t', '*'*(cont//1))
    
    #print(lens)
    ct_nonzero = 0
    for l in lens:
        if l == 0: ct_nonzero += 1
    
    return lens



def draw_all_histograms(type):
    print(f'Histograms of type={type}')
    print('-------------------------------')

    zeros = []
    
    for i in range(1, 600, 30):
        img_name = get_file_name(type, i)
        print(img_name)
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32,32))
        zeros.append(draw_histogram(img, bin_width=16))

        print('-------------------------------')
        print('\n')
    
    print(sorted(zeros))

# draw_all_histograms(NO_STICK_DATATYPE)

cap = cv2.VideoCapture(0)
drum_area = DrumArea(top_left_corner=(50, 50), square_dim=300, sound='j')
area_listener = AreaListener(drum_areas=[drum_area])

orig = None

while True:
    _, frame_orig = cap.read()
    frame_orig = cv2.flip(frame_orig, 1)
    frame_color = frame_orig.copy()
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)

    frame = np.asarray(frame)
    
    

    area_listener.draw_area(frame)
    ar = drum_area.get_area(frame)

    ar = cv2.resize(ar, (32,32))

    if orig is None:
        orig = ar

    diff = orig - ar

    diff_copy = diff.copy()
    for i in range(len(diff_copy)):
        for j in range(len(diff_copy[0])):
            if diff_copy[i][j] < 0:
                diff_copy[i][j] = 255


    cv2.imshow('Show', np.asarray(diff_copy))
    
    


    key = cv2.waitKey(1)
    if key == ord('s'):
        draw_histogram(ar, bin_width = 16)

    cv2.imshow('Main', frame)

    if key & 0xFF == ord('q'):
        break
