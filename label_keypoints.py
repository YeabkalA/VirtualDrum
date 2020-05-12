"""
Paul Devlin.

How to use: for each image, click on the keypoints, then press any key on your keyboard to move to the next image.
Your progress is saved as you go along. You can quit at any time by terminating the program and when you re-run the
program it will pick up where you left off.
The labels are saved in label_file as a string representation of a python dict.
"""
import os
from typing import Dict, List, Tuple
from tensor_sample_collector import ROOT_SAMPLES_DIR, LABEL_FILE, STICK_TIP_SAMPLE,BLURRY_STICK_TIP_SAMPLE, get_file_name

import cv2
import numpy as np
IM_SIZE= 320

class KeypointLabeler:
    def __init__(self, data_type, label_file):
        self.image_directory: str = f'{ROOT_SAMPLES_DIR}/{data_type}/'
        self.label_file: str = f'{ROOT_SAMPLES_DIR}/{label_file}'
        print('Labels file', label_file)
        self.keypoint_radius: int = 4
        self.current_filename: str = ""
        self.current_image = None
        self.labels = {}
        labels_string = self.read_labels().strip()
        if labels_string:
            self.labels = eval(labels_string)
    
    def num_labeled_imgs(self):
        return len(eval(self.read_labels().strip()))
    
        labels_string: str = self.read_labels().strip()
        if not labels_string:
            self.labels = {}
        else:
            self.labels = eval(labels_string)

    def save_label(self, event: int, x: int, y: int, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            curr_img_copy = self.current_image.copy()
            cv2.circle(curr_img_copy, (x, y), self.keypoint_radius, (255, 0, 0), -1)
            cv2.imshow("label_me_please", curr_img_copy)
            self.labels[self.current_filename] = (x, y)
            print((x,y))

    def label(self):
        print('found already labeled files ', len(self.labels))
        cv2.namedWindow("label_me_please")
        for filename in os.listdir(self.image_directory):
            print(filename)
            self.current_filename = filename
            if (self.current_filename in self.labels) and (self.labels[self.current_filename] != (0,0)):
                continue
            self.labels[self.current_filename] = (0,0)
            cv2.setMouseCallback("label_me_please", self.save_label)
            self.current_image = cv2.imread(self.image_directory + "/" + filename)
            cv2.imshow("label_me_please", self.current_image)
            key = cv2.waitKey(0)
            if key == ord('q'):
                return
            self.write_labels()

    def read_labels(self):
        if not os.path.exists(self.label_file):
            print('Nah')
            return ""
        with open(self.label_file) as label_output_file:
            label_string = label_output_file.readline()
        return label_string

    def write_labels(self):
        with open(self.label_file, "w") as label_output_file:
            label_output_file.write(str(self.labels))
    
    def reset_map(self):
        map_str = self.read_labels()
        print(map_str)
        orig_map = eval(map_str)
        new_map ={}
        for k in orig_map.keys():
            #new_map[f'stick_tip_samples/stick_tip_sample/{k}'] = orig_map[k]
            new_map[f'stick_tip_samples/blurry_stick_tip_sample/{k}'] = orig_map[k]
        with open(f'{ROOT_SAMPLES_DIR}/labels_main_blurry', "w") as label_output_file:
            label_output_file.write(str(new_map))
    
    def slide_show_labels(self, data_type):
        labels_str = self.read_labels()
        labels_map = eval(labels_str)
        for i in range(1,2001,1):
            file_name = get_file_name(data_type, i)
            img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            center = labels_map[file_name]
            cv2.circle(img, center, 8, (255, 0, 0), -1)
            cv2.imshow('Show', img)
            cv2.waitKey(10)
    
    def show_coverage(self, data_type):
        covered_points = set()
        labels_str = self.read_labels()
        labels_map = eval(labels_str)
        blank_img = np.zeros((IM_SIZE,IM_SIZE,3), np.uint8)
        for center in labels_map.values():
            cv2.circle(blank_img, center, 2, (30, 110, 100), -1)
            covered_points.add(center)
        print(len(covered_points)/(320**2))
        cv2.imshow('Show', blank_img)
        cv2.imwrite('stick_tip_coverage_1.9perc.jpg', blank_img)
        cv2.waitKey(0)
        




if __name__ == "__main__":
    keypoint_labeler = KeypointLabeler(label_file='labels_main_blurry', data_type=BLURRY_STICK_TIP_SAMPLE)
    keypoint_labeler.label()
    
    keypoint_labeler.reset_map()