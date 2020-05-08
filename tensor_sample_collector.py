import cv2
from area_listener import AreaListener, DrumArea
from image_processor import ImageProcessor
from ml_kit import sectional_density
import numpy as np
from os import listdir
from os.path import isfile, join
import re
from ml_kit import NN
from utils import get_saved_model
import tensorflow as tf
from tensorflow import keras
from sound_player import SoundPlayer
import utils
import os
import random


SQUARE_DIM_RESIZE = 80
RESIZE_DIM = (SQUARE_DIM_RESIZE, SQUARE_DIM_RESIZE)

DIVIDER = '======================='
SAMPLES_TO_COLLECT = 2092
# Images with no sticks; clear background. Used solely for testing.
SQUARE_DIM = 320
ROOT_SAMPLES_DIR = 'preprocessed/sample_80'

TEST_NO_STICK_DATATYPE = 'test_nostick_sample'
# Images with no sticks; clear background.
NO_STICK_DATATYPE = 'nostick_sample'
NO_STICK_DATATYPE_RESERVE = 'nostick_sample_reserve'
# Images with still sticks. Used solely for testing.
TEST_STICK_DATATYPE = 'test_stick_sample'
# Images with still sticks.
STICK_DATATYPE = 'stick_sample'
# Images with stick and other backgrounds
STICK_WITH_BACKGROUND_DATATYPE = 'stick_sample_with_bg'
# Background mixed with no stick
NO_STICK_BUT_BACKGROUND_DATATYPE = 'nostick_sample_but_with_bg'
# Images with blurry (moving) sticks.
BLURRY_DATATYPE = 'blurry_sample'
# Images that do not have sticks in them, but could have other objects.
NON_CLEAR_BACKGROUND_DATATYPE = 'non_clear_background_datatype'
# Images with sticks on the edge of the image area.
EDGE_STICK_DATATYPE = 'edge_stick_sample'

FAR_STICK_IMAGES = 'far_stick_sample'

PURELY_BLURRY_IMAGES = 'purely_blurry_sample'

NO_STICK_WHITENESS_ADDED = 'no_stick_whiteness_added_sample'

def weighted_sectional_density(img, make2D):
    return sectional_density(image=img, draw=False, w=4, h=4, make2D=make2D)

def darkness_gradient(img):
    flattened = np.array(img).flatten()
    flattened = sorted(flattened)
    return flattened[:10] + flattened[-10:]

def collect_preprocessed_samples(data_type, num_samples, saveOnKey = False, add_random_whiteness=False):
    s_activated = False

    directory = f'{ROOT_SAMPLES_DIR}/{data_type}/'
    os.system(f'mkdir {directory}')
    last_file_name = get_last_file_name(directory)
    cap = cv2.VideoCapture(0)
    
    drum_area = DrumArea(top_left_corner=(50, 50), square_dim=SQUARE_DIM, sound='j')
    #drum_area2 = DrumArea(top_left_corner=(800, 50), square_dim=SQUARE_DIM, sound='j')
    area_listener = AreaListener(drum_areas=[drum_area])
    img_process = ImageProcessor()

    count = last_file_name - 1
    base_set = False
    base_imgs = None

    max_black = 0
    while True:
        _, frame = cap.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = img_process.horizontal_flip(frame)

        targeted_area = area_listener.get_all_target_areas(img=frame, resize_dim=RESIZE_DIM)[0]
        area_listener.draw_area(frame)
        cv2.imshow('Target', targeted_area)
        cv2.imshow('Main', frame)

        if not base_set:
            area_listener.set_base_image(frame)
            base_imgs = area_listener.get_base_imgs(resize_dim=RESIZE_DIM)
            base_set = True
        
        diff = cv2.absdiff(targeted_area, base_imgs[0])
        diff_gray = np.asarray(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY))
        
        if add_random_whiteness:
            base = random.randint(0,10)
            for i in range(RESIZE_DIM[0]):
                for j in range(RESIZE_DIM[0]):
                    diff_gray[i][j] += (base + random.randint(0,3))
            
        cv2.imshow(f'diff_abs', diff_gray)
        #print(diff_gray)
        diff_gray = np.asarray(diff_gray)
        diff_gray_flat = diff_gray.flatten()

        key = cv2.waitKey(1)
        if key == ord('s'):
            if not s_activated:
                print('S activated at max_black =', max_black)
            s_activated = True
            count = last_file_name + 1
        
        if count > last_file_name and count <= (num_samples + last_file_name):
            if saveOnKey and key != ord('a'):
                continue
            # diff_gray[diff_gray > max_black] = 255
            # diff_gray[diff_gray <= max_black] = 0
            file_name = get_file_name(data_type, count)
            cv2.imwrite(file_name, diff_gray)
            count += 1
            print(f'Saved {file_name}')
        elif count > num_samples + last_file_name:
            return
        else:
            #print('chilling...')
            max_black = max(max_black, max(diff_gray_flat))

        if key & 0xFF == ord('q'):
            break
        
def collect_samples(data_type, num_samples, saveOnKey = False):
    s_activated = False

    directory = f'{ROOT_SAMPLES_DIR}/{data_type}/'
    os.system(f'mkdir {directory}')
    last_file_name = get_last_file_name(directory)
    cap = cv2.VideoCapture(0)
    
    drum_area = DrumArea(top_left_corner=(50, 50), square_dim=SQUARE_DIM, sound='j')
    #drum_area2 = DrumArea(top_left_corner=(800, 50), square_dim=SQUARE_DIM, sound='j')
    area_listener = AreaListener(drum_areas=[drum_area])
    img_process = ImageProcessor()

    count = last_file_name - 1

    while True:
        _, frame = cap.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = img_process.horizontal_flip(frame)

        targeted_area = area_listener.get_all_target_areas(img=frame)[0]
        area_listener.draw_area(frame)
        cv2.imshow('Target', targeted_area)
        cv2.imshow('Main', frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            if not s_activated:
                print('S activated')
            s_activated = True
            count = last_file_name + 1
        
        if count > last_file_name and count <= (num_samples + last_file_name):
            if saveOnKey and key != ord('a'):
                continue
            #file_name = get_file_name(data_type, count)
            file_name = 'mask_images/stick2.jpg'
            cv2.imwrite(file_name, targeted_area)
            count += 1
            print(f'Saved {file_name}')
        elif count > num_samples + last_file_name:
            return

        if key & 0xFF == ord('q'):
            break

# Not currently working
def test_model_live(model):
    cap = cv2.VideoCapture(0)
    drum_area2 = DrumArea(top_left_corner=(100, 100), square_dim=SQUARE_DIM, sound='j')
    area_listener2 = AreaListener(drum_areas=[drum_area2])
    img_process2 = ImageProcessor()
    probability_model2 = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    while True:
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = img_process2.horizontal_flip(frame)

        targeted_area = area_listener2.get_all_target_areas(img=frame)[0]
        area_listener2.draw_area(frame)
        
        cv2.imshow('Main', frame)

        # if prediction == 0:
        #     sp.play_key(ord(drum_area.sound))

        key = cv2.waitKey(1)
        if key == ord('s'):
            file_name = 'tp.jpg'
            cv2.imwrite(file_name, targeted_area)
            to_predict_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            to_predict_list = np.asarray([np.asarray(to_predict_img)])
            prediction, _ = utils.predict(to_predict_list[0], probability_model2, preprocess=False)
            cv2.imshow('Target', to_predict_img)
            print(f'PREDICTION={prediction}')

        if key & 0xFF == ord('q'):
            break
        
def test_weighted_sectional_density():
    sample_img_nostick = cv2.imread('nostick_sample/sample_nostick_sample2078.jpg', cv2.IMREAD_GRAYSCALE)
    sample_img_yastick = cv2.imread('stick_sample/sample_stick_sample5.jpg', cv2.IMREAD_GRAYSCALE)
    sd_nostick = weighted_sectional_density(sample_img_nostick)
    sd_yastick = weighted_sectional_density(sample_img_yastick)
    print(sd_nostick)
    print(sd_yastick)

def get_last_file_name(directory):
    onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    max_val = 0
    for file_name in onlyfiles:
        try:
            name = re.sub('[^0-9]', '', file_name).strip()
            max_val = max(max_val, int(name))
        except Exception:
            pass
    
    return max_val

def get_file_name(data_type, count):
    return f'{ROOT_SAMPLES_DIR}/{data_type}/sample_{data_type}{count}.jpg'

def get_num_stick_samples():
    return get_last_file_name(f'{ROOT_SAMPLES_DIR}/{STICK_DATATYPE}/')

def get_num_nostick_samples():
    return get_last_file_name(f'{ROOT_SAMPLES_DIR}/{NO_STICK_DATATYPE}/')

def get_num_test_stick_samples():
    return get_last_file_name(f'{ROOT_SAMPLES_DIR}/{TEST_STICK_DATATYPE}/')

def get_num_test_no_stick_samples():
    return get_last_file_name(f'{ROOT_SAMPLES_DIR}/{TEST_NO_STICK_DATATYPE}/')

def get_num_blurry_stick_samples():
    return get_last_file_name(f'{ROOT_SAMPLES_DIR}/{BLURRY_DATATYPE}/')

def print_training_and_test_data_status():
    print('num stick samples =', get_num_stick_samples())
    print(DIVIDER)
    #print('num test stick samples =', get_num_test_stick_samples())
    print(DIVIDER)
    print('num no stick samples =', get_num_nostick_samples())
    print(DIVIDER)
    #print('num test no stick samples =', get_num_test_no_stick_samples())
    print(DIVIDER)
    print('num blurry samples =', get_num_blurry_stick_samples())


def create_multiple_images(img):
    img = cv2.resize(img, RESIZE_DIM)
    images = []

    for rotate_factor in [cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, -190]:
        if rotate_factor == -190:
            img2 = img.copy()
        else:
            img2 = cv2.rotate(img, rotate_factor)
        images.append(img2)

    return images

def test_augmentations(img):
    augmentations = create_multiple_images(img)
    cv2.imshow('ORIG', img)
    cv2.waitKey(0)
    ct = 0
    for aug in augmentations:
        cv2.imshow(f'aug{ct}', aug)
        cv2.waitKey(0)
        ct += 1
    
if __name__ == '__main__':
    #test_model_live(get_saved_model('models/main_model'))
    #print(get_last_file_name('stick_sample/'))
    #test_weighted_sectional_density()
    #collect_preprocessed_samples(BLURRY_DATATYPE, 51, saveOnKey=True, add_random_whiteness=False)
    #print_training_and_test_data_status()
    test_augmentations(cv2.imread('preprocessed/sample_80/blurry_sample/sample_blurry_sample490.jpg', cv2.IMREAD_GRAYSCALE))