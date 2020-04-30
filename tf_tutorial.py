# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import cv2
import random
from tensor_sample_collector import weighted_sectional_density
from tensor_sample_collector import get_num_nostick_samples, get_num_stick_samples, get_num_test_stick_samples, get_num_test_no_stick_samples, get_num_blurry_stick_samples, get_file_name, get_last_file_name
from tensor_sample_collector import test_model_live
from area_listener import AreaListener, DrumArea
from image_processor import ImageProcessor
import utils
import random
import tensor_sample_collector

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def create_multiple_images(img):
    images = []

    for flip_factor in [-1, 0, 1]:
        img1 = cv2.flip(img, flip_factor)
        for rotate_factor in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            img2 = cv2.rotate(img1, rotate_factor)
            images.append(img2)

    return images

def split_to_test_train(data, train_perc):
    train_set, test_set = [], []
    perc10 = len(data) // 10
    i = 0
    for d in data:
        if i % perc10 == 0:
            print(f'Finished splitting {round(i*100/len(data), 3)}% of data')
        if random.randint(0, 101) <= train_perc:
            train_set.append(d)
        else:
            test_set.append(d)
        i += 1
    
    return train_set, test_set

def test_blurry():
    rv = []

    num_test_stick_samples = get_num_blurry_stick_samples()
    for i in range(1, num_test_stick_samples + 1):
        file_name = f'blurry_sample/sample_blurry_sample{i}.jpg'
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        rv.append((np.asarray(img), 0))
    
    images, labels = [],[]
    for data in rv:
        images.append(data[0])
        labels.append(data[1])
    
    return np.asarray(images), np.asarray(labels)


def test_specific_images():
    rv = []

    num_test_stick_samples = get_num_test_stick_samples()
    for i in range(1, num_test_stick_samples + 1):
        file_name = f'test_stick_sample/sample_test_stick_sample{i}.jpg'
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        rv.append((np.asarray(img), 0))
    
    num_test_no_stick_samples = get_num_test_no_stick_samples()
    for i in range(1, num_test_no_stick_samples + 1):
        file_name = f'test_nostick_sample/sample_test_nostick_sample{i}.jpg'
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        rv.append((np.asarray(img), 1))

    random.shuffle(rv)

    images, labels = [],[]
    for data in rv:
        images.append(data[0])
        labels.append(data[1])
    
    return np.asarray(images), np.asarray(labels)

def read_directory(dir_name, label):
    num_samples = get_last_file_name(f'{dir_name}/')
    images = []
    perc10 = num_samples // 10

    for i in range(1, num_samples + 1):
        if i % perc10 == 0:
            read_percentage = round(i*100/num_samples, 3)
            print(f'Finished reading {read_percentage}% of {dir_name}')
        file_name = get_file_name(dir_name, i)
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        if len(img) == 0: # Image has been deleted
            continue
        expanded_images_set = create_multiple_images(img)
        for new_image in expanded_images_set:
            images.append(np.asarray(new_image))
    
    train_data, test_data = split_to_test_train(images, 80)
    train_labels, test_labels = [label] * len(train_data), [label] * len(test_data)
    print('Train images', len(train_data))
    print('Train labels', len(train_labels))
    print('Test images', len(test_data))
    print('Test labels', len(test_labels))
    print('--------------------------------')

    return train_data, test_data, train_labels, test_labels

def read_images_modularized(directory_data):
    train_images, train_labels, test_images, test_labels = [], [], [], []

    for dir_info in directory_data:
        dir_name, dir_label = dir_info
        dir_train_data, dir_test_data, dir_train_labels, dir_test_labels = read_directory(dir_name, dir_label)
        train_images += dir_train_data
        train_labels += dir_train_labels
        test_images += dir_test_data
        test_labels += dir_test_labels
    
    print('Train images', len(train_images))
    print('Train labels', len(train_labels))
    print('Test images', len(test_images))
    print('Test labels', len(test_labels))
    return np.asarray(train_images), np.asarray(train_labels), np.asarray(test_images), np.asarray(test_labels)


def read_images():
    directory_data = []
    directory_data.append((tensor_sample_collector.STICK_DATATYPE, 0))
    directory_data.append((tensor_sample_collector.NO_STICK_DATATYPE, 1))
    directory_data.append((tensor_sample_collector.BLURRY_DATATYPE, 0))

    return read_images_modularized(directory_data)

def create_nn():
    data = read_images()
    train_images, train_labels, test_images, test_labels = data
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(80, 80)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)
    _, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    model.save('models/main_model')

    return model, data
    
def test_nn(model, data, testName='unnamed'):
    print('TEST STARTED FOR DATASET:', testName, '...data set of size = ', len(data[0]))
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    _, test_acc = model.evaluate(data[0],  data[1], verbose=2)
    images, labels = data
    predictions = [x[0] for x in probability_model.predict(images)]

    print(tf.math.confusion_matrix(labels, predictions))
    print('\nTest accuracy:', test_acc)
    print('------------------------------------------------------')
    return

# model, data = create_nn()
# test_nn(model, (data[2], data[3]), run_one_by_one_with_key=False, test_with_tf=True)
# test_nn(model, test_specific_images(), run_one_by_one_with_key=False, test_with_tf=True)
#test_model_live(model)    

model = tf.keras.models.load_model('models/main_model')
data = read_images()
test_nn(model, (data[2], data[3]), testName='random test data')
test_nn(model, test_specific_images(), testName='test specific data')
test_nn(model, test_blurry(), testName='blurry')

'''
cap = cv2.VideoCapture(0)
drum_area = DrumArea(top_left_corner=(900, 100), square_dim=80, sound='j')
drum_area2 = DrumArea(top_left_corner=(100, 100), square_dim=80, sound='j')
dareas = [drum_area, drum_area2]
area_listener2 = AreaListener(drum_areas=dareas)
img_process2 = ImageProcessor()
probability_model2 = tf.keras.Sequential([model, tf.keras.layers.Softmax()])


while True:
    _, frame_orig = cap.read()
    frame_orig = img_process2.horizontal_flip(frame_orig)
    frame_color = frame_orig.copy()
    frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
    
    targeted_areas = area_listener2.get_all_target_areas(img=frame)
    area_id = 0
    area_listener2.draw_area(frame)
    area_listener2.draw_area(frame_color)
    for targeted_area in targeted_areas:
        area_id += 1
        file_name = f'tp{area_id}.jpg'
        cv2.imwrite(file_name, targeted_area)
        to_predict_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        to_predict_list = np.asarray([np.asarray(to_predict_img)])
        prediction, predicted_img = utils.predict(to_predict_list[0], probability_model2, preprocess=False)

        if prediction == 0:
            cv2.circle(frame_color, (500,100*area_id), 90, ((area_id * 30 + 430) % 255,(area_id * 100) % 255, (20 * area_id) % 255), -1)
            dareas[area_id - 1].playSound()

        cv2.imshow('Target', to_predict_img)
        print(f'PREDICTION={prediction}')

    

    # if prediction == 0:
    #     sp.play_key(ord(drum_area.sound))

    key = cv2.waitKey(1)
    if key == ord('s'):
        pass
        # file_name = 'tp.jpg'
        # cv2.imwrite(file_name, targeted_area)
        # to_predict_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        # to_predict_list = np.asarray([np.asarray(to_predict_img)])
        # prediction, predicted_img = utils.predict(to_predict_list[0], probability_model2, preprocess=False)

        # if prediction == 0:
        #     cv2.circle(frame, (500,500), 40, (0,0,0), -1)
        # cv2.imshow('Target', to_predict_img)
        # print(f'PREDICTION={prediction}')
    
    cv2.imshow('Main', frame_color)

    if key & 0xFF == ord('q'):
        break
'''