import cv2
import consts
import image_processor
import ml_kit
import numpy as np
import sound_player
import image_difference_tool
import math

class DrumArea(object):
    def __init__(self, top_left_corner, square_dim, sound):
        self.top_left_corner = top_left_corner
        self.square_dim = square_dim
        self.sound = sound
        self.bottom_left_corner = (self.top_left_corner[0] + self.square_dim, \
        self.top_left_corner[1] + self.square_dim)

class AreaListener(object):
    def __init__(self, drum_areas):
        self.drum_areas = drum_areas
        self.img_process = image_processor.ImageProcessor()
        self.sound_player = sound_player.SoundPlayer()
        self.img_dfc_tool = image_difference_tool.ImageDifferenceTool()
        self.prev_color_check = False
    
    def draw_area(self, img):
        for drum_area in self.drum_areas:
            cv2.rectangle(img, drum_area.top_left_corner, drum_area.bottom_left_corner, consts.RED, consts.AREA_BOUNDARY_THICKNESS) 
    
    def check_for_change(self):
        pass
        
    def get_area(self, img, drum_area):
        return img[drum_area.top_left_corner[1]:drum_area.bottom_left_corner[1], \
            drum_area.top_left_corner[0]:drum_area.bottom_left_corner[0]]
    
    def get_testable_img(self, img):
        return self.get_area(self.img_process.unsharp_mask(img))
    
    def number_of_edges(self, img):
        return len(self.img_process.Hough_lines(img))
    
    def set_base_image(self, img):
        # self.base_img = self.get_testable_img(img)
        # cv2.imwrite(image_difference_tool.BASE_IMG_DIR, self.base_img)
        self.base_imgs = []
        for drum_area in self.drum_areas:
            self.base_imgs.append(self.get_area(img, drum_area))
    
    def compare_difference_and_play_sound(self, frame):
        for index, drum_area in enumerate(self.drum_areas):
            area = self.get_area(frame, drum_area)
            diff = self.img_dfc_tool.ColorDiffVector(area, self.base_imgs[index])

            print(diff)
            
            color_check = diff > 1

            # print(diff)
            # color_check = self.img_dfc_tool.ColorDiff(area)

            if color_check:
                self.sound_player.play_key(ord(drum_area.sound))

        #     if self.prev_color_check == False:
        #         self.sound_player.play_key(ord('c'))
        #         self.prev_color_check = True
        # else:
        #     self.prev_color_check = False
       

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

    

    

        