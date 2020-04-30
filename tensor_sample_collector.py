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

SAMPLES_TO_COLLECT = 2092
# Images with no sticks; clear background. Used solely for testing.
TEST_NO_STICK_DATATYPE = 'test_nostick_sample'
# Images with no sticks; clear background.
NO_STICK_DATATYPE = 'nostick_sample'
# Images with still sticks. Used solely for testing.
TEST_STICK_DATATYPE = 'test_stick_sample'
# Images with still sticks.
STICK_DATATYPE = 'stick_sample'
# Images with blurry (moving) sticks.
BLURRY_DATATYPE = 'blurry_sample'
# Images that do not have sticks in them, but could have other objects.
NON_CLEAR_BACKGROUND_DATATYPE = 'non_clear_background_datatype'

def weighted_sectional_density(img, make2D):
    return sectional_density(image=img, draw=False, w=4, h=4, make2D=make2D)

def darkness_gradient(img):
    flattened = np.array(img).flatten()
    flattened = sorted(flattened)
    return flattened[:10] + flattened[-10:]

def collect_samples(data_type, num_samples, saveOnKey = False):
    last_file_name = get_last_file_name(f'{data_type}/')
    cap = cv2.VideoCapture(0)
    
    drum_area = DrumArea(top_left_corner=(100, 100), square_dim=80, sound='j')
    area_listener = AreaListener(drum_areas=[drum_area])
    img_process = ImageProcessor()

    count = last_file_name - 1

    while True:
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = img_process.horizontal_flip(frame)

        targeted_area = area_listener.get_all_target_areas(img=frame)[0]
        area_listener.draw_area(frame)
        cv2.imshow('Target', targeted_area)
        cv2.imshow('Main', frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            count = last_file_name + 1
        
        if count > last_file_name and count <= (num_samples + last_file_name):
            if saveOnKey and key != ord('a'):
                continue
            file_name = f'{data_type}/sample_{data_type}{count}.jpg'
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
    drum_area2 = DrumArea(top_left_corner=(100, 100), square_dim=80, sound='j')
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
    return f'{data_type}/sample_{data_type}{count}.jpg'

def get_num_stick_samples():
    return get_last_file_name('stick_sample/')

def get_num_nostick_samples():
    return get_last_file_name('nostick_sample/')

def get_num_test_stick_samples():
    return get_last_file_name(f'{TEST_STICK_DATATYPE}/')

def get_num_test_no_stick_samples():
    return get_last_file_name(f'{TEST_NO_STICK_DATATYPE}/')

def get_num_blurry_stick_samples():
    return get_last_file_name(f'{BLURRY_DATATYPE}/')

if __name__ == '__main__':
    #test_model_live(get_saved_model('models/main_model'))
    #print(get_last_file_name('stick_sample/'))
    #test_weighted_sectional_density()
    collect_samples(NON_CLEAR_BACKGROUND_DATATYPE, 200, saveOnKey=False)