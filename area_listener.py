import cv2
import consts
import image_processor
import ml_kit
import numpy as np
import sound_player
import image_difference_tool
import math
import random
from utils import get_saved_model
import tensorflow as tf
from ml_kit import NN
import time

class DrumArea(object):
    def __init__(self, top_left_corner, square_dim, sound):
        self.last_played = 0
        self.top_left_corner = top_left_corner
        self.square_dim = square_dim
        self.sound = sound
        self.bottom_left_corner = (self.top_left_corner[0] + self.square_dim, \
        self.top_left_corner[1] + self.square_dim)
        self.sp = sound_player.SoundPlayer()
        self.previous_beat_time = 0
        self.is_clear = True
        self.base_img = None
        self.id = 0

    def playSound(self):
        curr_millis = time.time() * 1000
        if curr_millis - self.previous_beat_time >= consts.REASONABLE_DRUM_BEAT_INTERVAL_MILLIS and self.is_clear:
            self.sp.play_key(ord(self.sound))
            self.previous_beat_time = curr_millis
    
    def capture(self):
        self.is_clear = False

    def free(self):
        self.is_clear = True
    
    def markPlayed(self, frame):
        cv2.rectangle(frame, self.top_left_corner, self.bottom_left_corner, consts.RED, consts.FILL_THICKNESS) 
    
    def get_area(self, img):
        area = img[self.top_left_corner[1]:self.bottom_left_corner[1], \
            self.top_left_corner[0]:self.bottom_left_corner[0]]
        return cv2.medianBlur(area, 5)

class AreaListener(object):
    def __init__(self, drum_areas):
        self.drum_areas = drum_areas
        for i in range(len(drum_areas)):
            self.drum_areas[i].id = i
        self.img_process = image_processor.ImageProcessor()
        self.sound_player = sound_player.SoundPlayer()
        self.img_dfc_tool = image_difference_tool.ImageDifferenceTool()
        self.prev_color_check = False
        self.nn = NN('models/main_model')
    
    # Draws each drum area in this AreaListener on the cv2 screen.
    def draw_area(self, img):
        for drum_area in self.drum_areas:
            random_color = (random.randint(0, 200), random.randint(0, 200), random.randint(0,200))
            cv2.rectangle(img, drum_area.top_left_corner, drum_area.bottom_left_corner, consts.RED, consts.AREA_BOUNDARY_THICKNESS) 
    
    def check_for_change(self):
        pass
        
    # Gets the part of `img` in the given `drum_area`.
    def get_area(self, img, drum_area):
        return img[drum_area.top_left_corner[1]:drum_area.bottom_left_corner[1], \
            drum_area.top_left_corner[0]:drum_area.bottom_left_corner[0]]
    
    # Does the preprocessing of `img` to make it ready for testing.
    def get_testable_img(self, img):
        return self.get_area(self.img_process.unsharp_mask(img))
    
    # Observed to be a weak measure, since the number of edge is arbitrary.
    # The addition of two edges does not create a signficant number of 
    # edges to the drum area.
    def number_of_edges(self, img):
        return len(self.img_process.Hough_lines(img))
    
    def get_all_target_areas(self, img, resize_dim=None):
        rv = []
        for drum_area in self.drum_areas:
            target_area = drum_area.get_area(img)
            if resize_dim:
                target_area = cv2.resize(target_area, resize_dim)
            target_area = cv2.medianBlur(target_area, 5)
            rv.append(target_area)
            drum_area.base_img = target_area
        
        return rv
    
    def set_base_image(self, img):
        # self.base_img = self.get_testable_img(img)
        # cv2.imwrite(image_difference_tool.BASE_IMG_DIR, self.base_img)
        self.base_imgs = self.get_all_target_areas(img)
    
    def get_base_imgs(self, resize_dim=None):
        print('Resize dim', resize_dim)
        if resize_dim:
            return [cv2.resize(x, resize_dim) for x in self.base_imgs]
        else:
            return self.base_imgs
    
    def compare_difference_and_play_sound(self, frame):
        for index, drum_area in enumerate(self.drum_areas):
            area = self.get_area(frame, drum_area)
            diff = self.img_dfc_tool.ColorDiffVector(area, self.base_imgs[index])

            print(diff)
            color_check = diff > 30

            if color_check:
                self.sound_player.play_key(ord(drum_area.sound))
    
    def check_through_nn(self, frame):
        for index, drum_area in enumerate(self.drum_areas):
            area = self.get_area(frame, drum_area)
            area_bw = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
            prediction = self.nn.predict(area_bw)
            cv2.imwrite('predicted.jpg', area_bw)
            print(prediction)
            if prediction == 0:
                self.sound_player.play_key(ord(drum_area.sound))

            


       

        # # Through ImageChops
        # curr_testable_img = self.get_testable_img(frame)
        # cv2.imwrite(image_difference_tool.NEW_IMG, curr_testable_img)
        # self.img_dfc_tool.DiffThroughImageChops()
        # diff_img = cv2.imread(image_difference_tool.DIFF_IMG_DIR, cv2.IMREAD_GRAYSCALE)
        # diff_sd = ml_kit.sectional_density(diff_img)
        # diff = np.linalg.norm(diff_sd)
        # print(diff)
        
        # diff = np.linalg.norm(np.asarray(ml_kit.sectional_density(curr_testable_img)) - np.asarray(ml_kit.sectional_density(self.base_img)))
        # print(diff)
        # # cv2.imwrite('curr_test.jpg', curr_testable_img)
        # # diff = self.img_dfc_tool.StructuralSimilarityIndex(self.base_img, curr_testable_img)
        # # num_lines_diff = abs(self.number_of_edges(self.base_img) - self.number_of_edges(curr_testable_img))
        # # print(f'Diff={diff}, num_lines_diff = ({num_lines_diff})')
        # if  diff > 7000:
        #     self.sound_player.play_key(ord('j'))
    

        