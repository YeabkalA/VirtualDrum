import cv2
import consts
import time
from utils import get_saved_model
import tensorflow as tf
from tensorflow import keras
import numpy as np

def sectional_density(image, draw=False, sparsity=1, w=4, h=4, weighted=False, make2D = False):
    steps = 0
    image_size = len(image)

    CELL_WIDTH, CELL_HEIGHT = w, h
    pixel_percentages = [0 for i in range((image_size // CELL_WIDTH) * (image_size // CELL_HEIGHT))]
    total_black_pixels, count = 0, 0

    for corner_y in range(0, (image_size - CELL_HEIGHT + 1), CELL_HEIGHT):
        for corner_x in range(0, (image_size - CELL_WIDTH + 1), CELL_WIDTH):
            if draw:
                cv2.rectangle(image, (corner_x, corner_y), (corner_x+CELL_WIDTH, corner_y+CELL_HEIGHT), \
                     consts.RED, consts.AREA_BOUNDARY_THICKNESS) 
            for i in range(0, CELL_HEIGHT, sparsity):
                for j in range(0, CELL_WIDTH, sparsity):
                    steps += 1
                    pixel_percentages[count] += image[corner_y + i][corner_x + j]
                    total_black_pixels += image[corner_y + i][corner_x + j]
            count += 1
    
    # if weighted:
    #     for i in range(len(pixel_percentages)):
    #         pixel_percentages[i] = pixel_percentages[i] * total_black_pixels

    if make2D:
        pixel_percentages = [pixel_percentages[i:i+w] for i in [w*j for j in range(w)]]

    return pixel_percentages


class NN(object):
    def __init__(self, model_name):
        self.model = get_saved_model(model_name)
        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        
    def predict(self, img):
        bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(bw_img, (80, 80))
        unknown_img = np.array([resized_img]).astype(np.float32)
        prediction = self.probability_model.predict_classes(unknown_img)[0]

        return prediction