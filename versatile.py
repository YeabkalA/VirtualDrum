from area_listener import AreaListener, DrumArea
from image_processor import ImageProcessor
import cv2
from image_difference_tool import ImageDifferenceTool
import numpy as np
#from multiprocessor import Multiprocessor

SQUARE_DIM = 120
RESIZE_DIM = (SQUARE_DIM, SQUARE_DIM)
def test_live_effecient(square_dim=320):
    cap = cv2.VideoCapture(0)
    img_process2 = ImageProcessor()

    base_set = False
    base_imgs = None

    
    test_max_black_pixel_count = 0
    drum_area = DrumArea(top_left_corner=(100,10), square_dim=square_dim, sound='j')
    drum_area2 = DrumArea(top_left_corner=(100,320), square_dim=square_dim, sound='c')
    drum_areas = [drum_area, drum_area2]
    #drum_areas = [drum_area2]

    # drum_areas = []
    # for i in range(0, 900, 300):
    #     for j in range(0, 900, 300):
    #         drum_areas.append(DrumArea(top_left_corner=(i,j), square_dim=square_dim, sound='c'))
    area_listener = AreaListener(drum_areas = drum_areas)
    last_states = [False for i in range(len(drum_areas))]
    max_black_pixel = [0 for i in range(len(drum_areas))]

    while True:
        _, frame_orig = cap.read()
        frame_orig = img_process2.horizontal_flip(frame_orig)
        area_listener.draw_area(frame_orig)
        
        if not base_set:
            area_listener.set_base_image(frame_orig)
            base_imgs = area_listener.get_base_imgs(resize_dim=RESIZE_DIM)
            base_set = True
    
        target_areas = area_listener.get_all_target_areas(frame_orig, resize_dim=RESIZE_DIM)
        for i,ta in enumerate(target_areas):
            diff = cv2.absdiff(ta, base_imgs[i])
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            cv2.imshow(f'diff_abs{i}', diff_gray)
            #print(diff_gray)
            diff_gray = np.asarray(diff_gray)
            diff_gray_flat = diff_gray.flatten()
            if test_max_black_pixel_count < 100:
                max_black_pixel[i] = max(max_black_pixel[i], max(diff_gray_flat))
            else:
                diff_gray[diff_gray > max_black_pixel[i]+10] = 255
                diff_gray[diff_gray <= max_black_pixel[i]+10] = 0
                #cv2.circle(frame_orig, (500,100*(i+1)), 90, (100,140,10), -1)
                cv2.imshow(f'diff{i}', diff_gray)

                num_whites = len(diff_gray[diff_gray == 255])
                print((i, num_whites), max_black_pixel)
                if num_whites > 2:
                    if not last_states[i]:
                        last_states[i] = True
                        drum_areas[i].playSound()
                        drum_areas[i].markPlayed(frame_orig)
                else:
                    last_states[i] = False
                
            cv2.waitKey(1)
        test_max_black_pixel_count += 1
        
        cv2.imshow('Main', frame_orig)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break    

def test_live():
    # if True:
    #     return
    cap = cv2.VideoCapture(0)

    drum_area = DrumArea(top_left_corner=(50, 50), square_dim=320, sound='c')
    drum_area2 = DrumArea(top_left_corner=(800, 50), square_dim=160, sound='j')
    area_listener2 = AreaListener(drum_areas=[drum_area2])
    img_process2 = ImageProcessor()
    imDfTool = ImageDifferenceTool()

    base_set = False
    base_imgs = None
    drum_areas_set = False

    max_perc = -1
    max_black_pixel = 0
    test_max_black_pixel_count = 0
    drum_area = DrumArea(top_left_corner=(0,0), square_dim=100, sound='j')
    while True:
        _, frame_orig = cap.read()
        if not drum_areas_set:
            area_listener2.drum_areas = [drum_area]
            drum_areas_set = True

        frame_orig = img_process2.horizontal_flip(frame_orig)
        area_listener2.draw_area(frame_orig)
        
        if not base_set:
            area_listener2.set_base_image(frame_orig)
            base_imgs = area_listener2.get_base_imgs()
            # for i, bi in enumerate(base_imgs):
            #     cv2.imshow(f'base{i}', bi)
            #     cv2.waitKey(1)
            base_set = True
        
        target_areas = area_listener2.get_all_target_areas(frame_orig)
        diffs = []
        #https://stackoverflow.com/questions/27035672/cv-extract-differences-between-two-images
        
        
        for i,ta in enumerate(target_areas):
            diff = cv2.absdiff(ta, base_imgs[i])
            mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            th = 1
            imask =  mask>th
            #print(imask)

            canvas = np.zeros_like(ta, np.uint8)
            canvas[imask] = ta[imask]

            # mask = np.asarray(mask)
            # mask[mask>]
            
            # orig = cv2.cvtColor(base_imgs[i], cv2.COLOR_BGR2GRAY)
            # #rig = cv2.threshold(orig,100,255,cv2.THRESH_BINARY)[1]
            # new = cv2.cvtColor(ta, cv2.COLOR_BGR2GRAY)
            #new = cv2.threshold(new,127,255,cv2.THRESH_BINARY)[1]

            #img, mask = imDfTool.ColorDiffRob(orig, new)
            #mask = cv2.medianBlur(mask, 5)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            diff_gray = np.asarray(diff_gray)
            diff_gray_flat = diff_gray.flatten()
            average = np.sum(diff_gray_flat)/len(diff_gray_flat)
            if test_max_black_pixel_count < 100:
                max_black_pixel = max(max_black_pixel, max(diff_gray_flat))
            else:
                diff_gray[diff_gray > max_black_pixel+10] = 255
                diff_gray[diff_gray <= max_black_pixel+10] = 0
                cv2.circle(frame_orig, (500,100), 90, (100,140,10), -1)
                if len(diff_gray[diff_gray == 255] > 2):
                    drum_area.playSound()
                
            test_max_black_pixel_count += 1

            print(average, 'average', max(diff_gray_flat), 'max')
            # diff_gray[diff_gray > 35] = 255
            # cv2.imshow(f'canvase{i}', canvas)
            cv2.imshow(f'diff_gray{i}', diff_gray)
            # cv2.imshow(f'orig{i}', base_imgs[i])
            # cv2.imshow(f'target{i}', ta)

            # perc = imDfTool.GetWhitePercentage(mask)
            # max_perc = max(max_perc, perc)

            # cv2.imshow(f'orig{i}', orig)
            cv2.waitKey(1)
            # cv2.imshow(f'new{i}', new)
            # cv2.waitKey(1)

            # print(perc, max_perc)
            # if perc > 0.0:
            #     drum_area2.playSound()
            #     area_id = 2
            #     cv2.circle(frame_orig, (500,100*area_id), 90, ((area_id * 30 + 430) % 255,(area_id * 100) % 255, (20 * area_id) % 255), -1)
            #     #drum_area2.markPlayed(frame_orig)
            #     drum_area2.is_clear = False
            # else:
            #     drum_area2.is_clear = True
        
        cv2.imshow('Main', frame_orig)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

test_live_effecient(300)
#Multiprocessor()