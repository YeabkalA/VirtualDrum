import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import os
import random
import time
from multiprocessing import Process, Queue, current_process
import matplotlib.pyplot as plt
import geom

from area_listener import AreaListener, DrumArea
from image_processor import ImageProcessor

LABELS_FILE_NAME = 'stick_tip_samples/labels_main'
LABELS_FILE_NAME_BLURRY = 'stick_tip_samples/labels_main_blurry'

#ALL_LABELS_FILES = [LABELS_FILE_NAME, LABELS_FILE_NAME_BLURRY]
ALL_LABELS_FILES = [LABELS_FILE_NAME]

tf.get_logger().setLevel('INFO')

def get_augmentations(img, center):
    rotation_types = [cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180]
    rotated_img_center_pairs = [(cv2.rotate(img.copy(), i), geom.rotate(i, center, 320)) for i in rotation_types]

    flip_types = [0,1]
    flipped_img_center_pairs = [(cv2.flip(img.copy(), i), geom.flip(i, center, 320)) for i in flip_types]

    return flipped_img_center_pairs + rotated_img_center_pairs + [(img, center)]

def get_augmentations_with_brightness_adjustments(img, center, dim):
    augs = []

    non_brightness_augs = get_augmentations(img, center) + get_augmentations(img, (center[0] + 2, center[1] + 0)) + get_augmentations(img, (center[0] - 2, center[1] + 0)) + get_augmentations(img, (center[0] + 0, center[1] - 2)) + get_augmentations(img, (center[0] - 0, center[1] + 2)) 
    for nba_data in non_brightness_augs:
        nba = nba_data[0]
        for bright in range(5,30,5):
            bright_delta = bright/100
            brightness_adjusted = brightness_tf(nba, bright_delta, dim)
            augs.append((brightness_adjusted, nba_data[1]))
    
    return augs

def test_augmentation():
    img = cv2.imread('stick_tip_samples/stick_tip_sample/sample_stick_tip_sample1446.jpg', cv2.IMREAD_GRAYSCALE)
    center = (119, 163)
    img = cv2.resize(img, (320,320))
    augmentations = get_augmentations_with_brightness_adjustments(img, center, 320)
    print('Len augmentations', len(augmentations))
    ct = 0
    for augmentation_data in augmentations:
        aug_img, aug_center = augmentation_data
        aug_color = cv2.cvtColor(aug_img, cv2.COLOR_GRAY2BGR)
        cv2.circle(aug_color, aug_center, 8, (0,255,0), -1)
        aug_color = cv2.resize(aug_color, (120,120))
        cv2.imshow(f'Aug{ct}', aug_color)
        cv2.imwrite(f'Aug{ct}.jpg', aug_color)
        ct+=1
        #cv2.waitKey(0)

def read_labels(resize_dim=80):
    data_list = []
    for a_label_file in ALL_LABELS_FILES:
        file_center_map = eval(open(a_label_file, 'r').read().strip())
    
        count = 0
        for file in file_center_map.keys():
            if count % 50 == 0:
                print(f'Read {count} files')
            count += 1
            
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (resize_dim, resize_dim))
            center = file_center_map[file]
            augs = get_augmentations_with_brightness_adjustments(img, center, resize_dim)
            augs = [(cv2.resize(x[0], (resize_dim, resize_dim)), x[1]) for x in augs]
            
            for aug in augs:
                data_list.append((tf.reshape(aug[0], [resize_dim, resize_dim, 1]).numpy(), aug[1]))
    
    print('About to create labels')
    train_data, train_label = [], []
    for data in data_list:
        train_data.append(data[0])
        train_label.append(data[1])
    print('Done to create labels')
    
    return train_data, train_label

def build_tip_detection_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(80, 80,1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='mse',
                  metrics=['mse'])
    train, train_labels = read_labels(80)
    train = np.asarray(train, dtype=np.float32)/ 255.0
    print('Read data. Going to train on', len(train), 'images')

    # Train the network
    print("Starting to train the network, on", len(train), "samples.")
    model.fit(np.asarray(train, dtype=np.float32), np.asarray(train_labels), \
              epochs=30, batch_size=8)
    model.save("models/light-augmentations-tip-detecting-with-random-point-aug")
    print("Done training network.")

# print(read_labels()[1])

def predict_for_image(file_name, model, i):
    print(file_name)
    img = cv2.imread(file_name)
    img_copy = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (80,80))
    img = tf.reshape(img, [80, 80, 1])
    center = model.predict(np.asarray([img.numpy()/255.0]))[0]
    cv2.circle(img_copy, tuple(map(int, center)), 9, (30, 110, 200), -1)
    cv2.imshow('Stick Tip Predictions', img_copy)
    wait_time = 0 if (i == 0 or i == 499) else 1
    cv2.waitKey(wait_time)

def brightness_tf(img, perc, dim):
    img = tf.reshape(img, [dim, dim, 1])
    img = tf.image.adjust_brightness(img, delta=perc)
    img = img.numpy()

    return img

def test_live(model):
    cap = cv2.VideoCapture(0)
    drum_area1 = DrumArea(top_left_corner=(50, 50), square_dim=320, sound='c')

    dareas = [drum_area1]
    area_listener = AreaListener(drum_areas=dareas)
    img_process = ImageProcessor()

    base_set = False
    base_imgs = None

    while True:
        _, frame_orig = cap.read()
        frame_orig = img_process.horizontal_flip(frame_orig)
        frame_color = frame_orig.copy()
        frame = frame_color.copy()

        if not base_set:
            area_listener.set_base_image(frame)
            base_imgs = area_listener2.get_base_imgs()
            base_set = True
        
        for drum_area in dareas:
            orig, target = drum_area.get_area(frame_orig), drum_area.base_img
            img_copy = orig.copy()
            diff = cv2.absdiff(target, orig)
            diff_gray = np.asarray(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY))

            diff_gray = cv2.resize(diff_gray, (80,80))
            diff_gray = tf.reshape(diff_gray, [80, 80, 1])
            center = model.predict(np.asarray([diff_gray.numpy()/255.0]))[0]
            cv2.circle(img_copy, tuple(map(int, center)), 9, (30, 110, 200), -1)
            cv2.imshow('Pred', img_copy)
            
            cv2.waitKey(1)
            
        area_listener2.draw_area(frame_color)
        key = cv2.waitKey(1)

        cv2.imshow('Main', frame_color)
        cv2.waitKey(1)

        if key == ord('s'):
            print('resetting base')
            area_listener2.set_base_image(frame)
            base_imgs = area_listener2.get_base_imgs()

        if key & 0xFF == ord('q'):
            break

def run_model_on_test_data(model):
    start = 1001
    for i in range(start,start + 500):
        img = f'preprocessed/sample_80/test_tip_sample/sample_test_tip_sample{i}.jpg'
        print(i)
        predict_for_image(img, model, i-start)

if __name__ == '__main__':
    model = tf.keras.models.load_model('models/light-augmentations-tip-detecting')
    run_model_on_test_data(model)