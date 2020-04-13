import cv2
import consts
import image_processor
import ml_kit
import numpy as np
import sound_player
import image_difference_tool

class AreaListener(object):
    def __init__(self, top_left_corner, square_dim):
        self.top_left_corner = top_left_corner
        self.square_dim = square_dim
        self.bottom_left_corner = (self.top_left_corner[0] + self.square_dim, \
            self.top_left_corner[1] + self.square_dim)
        self.img_process = image_processor.ImageProcessor()
        self.sound_player = sound_player.SoundPlayer()
        self.img_dfc_tool = image_difference_tool.ImageDifferenceTool()
    
    def draw_area(self, img):
        cv2.rectangle(img, self.top_left_corner, self.bottom_left_corner, consts.RED, consts.AREA_BOUNDARY_THICKNESS) 
    
    def check_for_change(self):
        pass
        
    def get_area(self, img):
        return img[self.top_left_corner[1]:self.bottom_left_corner[1], \
            self.top_left_corner[0]:self.bottom_left_corner[0]]
    
    def get_testable_img(self, img):
        return self.get_area(self.img_process.preprocess_image(img))
    
    def set_base_image(self, img):
        self.base_img = self.get_testable_img(img)
        cv2.imwrite('test_area.jpg', self.base_img)
    
    def compare_difference_and_play_sound(self, frame):
        curr_testable_img = self.get_testable_img(frame)
        cv2.imwrite('curr_test.jpg', curr_testable_img)
        diff = self.img_dfc_tool.StructuralSimilarityIndex(self.base_img, curr_testable_img)
        print(diff)
        if  diff > 150:
            self.sound_player.play_key(ord('j'))

    

    

        