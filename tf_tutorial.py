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
    # for flip_factor in [-1, 0, 1]:
    #     img0 = cv2.flip(img, flip_factor)
    #     for rotate_factor in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
    #         img1 = cv2.rotate(img0, rotate_factor)
    #         for bright in [0.0, 0.1, 0.2]:
    #             img3 = tf.image.adjust_brightness(img1, delta=bright)
    #             for blur_kern in [(1, 1), (1, 3), (3, 1), (3, 3), (5, 5)]:
    #                 img4 = cv2.blur(img3.numpy(), blur_kern)
    #                 images.append(img4)

    # return images

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

def test_blurry_on_no_blurry_samples():
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

    '''
    stick_images, nostick_images, blurry_stick_images = [], [], []
    num_stick_samples = get_num_stick_samples()
    num_nostick_samples = get_num_nostick_samples()
    num_blurry_samples = get_num_blurry_stick_samples()

    perc10_stick = num_stick_samples // 10
    perc10_nostick = num_nostick_samples // 10
    perc10_blurry = num_blurry_samples // 10

    for i in range(1, num_stick_samples + 1):
        if i % perc10_stick == 0:
            print(f'Finished reading {round(i*100/num_stick_samples, 3)}% of stick data')
        file_name = f'stick_sample/sample_stick_sample{i}.jpg'
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

        if len(img) == 0: # Image has been deleted
            continue
        expanded_images_set = create_multiple_images(img)
        for new_image in expanded_images_set:
            stick_images.append(np.asarray(new_image))

    for i in range(1, num_nostick_samples + 1):
        if i % perc10_nostick == 0:
            print(f'Finished reading {round(i*100/num_nostick_samples, 3)}% of no stick data')
        file_name = f'nostick_sample/sample_nostick_sample{i}.jpg'
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

        if len(img) == 0: # Image has been deleted
            continue
        expanded_images_set = create_multiple_images(img)
        for new_image in expanded_images_set:
            nostick_images.append(np.asarray(new_image))
        
    for i in range(1, num_blurry_samples + 1):
        if i % perc10_blurry == 0:
            print(f'Finished reading {round(i*100/num_nostick_samples, 3)}% of blurry data')
        file_name = f'blurry_sample/sample_blurry_sample{i}.jpg'
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

        if len(img) == 0: # Image has been deleted
            continue
        expanded_images_set = create_multiple_images(img)
        for new_image in expanded_images_set:
            blurry_stick_images.append(np.asarray(new_image))
    
    print(f'Found a total of {len(stick_images)} stick images and {len(nostick_images)} nostick images and {len(blurry_stick_images)} blurry images')
    stick_train, stick_test = split_to_test_train(stick_images, 80)
    nostick_train, nostick_test = split_to_test_train(nostick_images, 80)
    blurry_train, blurry_test = split_to_test_train(blurry_stick_images, 80)

    train_images = np.asarray(stick_train + nostick_train + blurry_train)
    train_labels = np.asarray(([0] * len(stick_train)) + ([1] * len(nostick_train)) + ([0] * len(blurry_train)))
    test_images = np.asarray(stick_test + nostick_test + blurry_test)
    test_labels = np.asarray(([0] * len(stick_test)) + ([1] * len(nostick_test)) + ([0] * len(blurry_test)))

    print(f'Using {len(train_images)}  images for training, and {len(test_images)} for testing.')
    print(f'Percentage used for testing = {len(test_images)/(len(train_images) + len(test_images))}')

    return train_images, train_labels, test_images, test_labels
    '''

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
    
def test_nn(model, data, run_one_by_one_with_key=False, test_with_tf=False, testName='unnamed'):
    if testName != 'unnamed':
        print(f'Test: {testName}')

    if test_with_tf:
        _, test_acc = model.evaluate(data[0],  data[1], verbose=2)
        print('\nTest accuracy:', test_acc, 'on data set of size', len(data[0]))
        return

    test_images, test_labels = data[0], data[1]
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    #predictions = probability_model.predict(test_images)

    success = 0.0
    failure = 0.0
    predictions = [[0 for i in range(2)] for j in range(2)]

    feature_map = {0:[], 1:[]}
    for i in range(len(test_images)):
        feature_map[test_labels[i]].append(test_images[i])

    for digit in range(2):
        if digit not in feature_map.keys():
            continue
        for f in feature_map[digit]:
            prediction = utils.predict(f, probability_model, preprocess=False)[0]
            predictions[digit][int(prediction)] += 1
            if run_one_by_one_with_key:
                cv2.imshow(f'Actual {digit} predicted {prediction}', f)
                cv2.waitKey(0)
            if prediction == digit:
                success += 1
            else:
                failure += 1
            
    print(success/(success + failure))
    print(np.array(predictions))

# orig_image = cv2.imread('stick_sample/sample_stick_sample5.jpg', cv2.IMREAD_GRAYSCALE)
# new_images = create_multiple_images(orig_image)
# print(len(new_images))




# model, data = create_nn()
# test_nn(model, (data[2], data[3]), run_one_by_one_with_key=False, test_with_tf=True)
# test_nn(model, test_specific_images(), run_one_by_one_with_key=False, test_with_tf=True)
#test_model_live(model)    

model = tf.keras.models.load_model('models/main_model')
data = read_images()
test_nn(model, (data[2], data[3]), run_one_by_one_with_key=False, test_with_tf=True, testName='random test data')
test_nn(model, test_specific_images(), run_one_by_one_with_key=False, test_with_tf=True, testName='test specific data')
test_nn(model, test_blurry_on_no_blurry_samples(), run_one_by_one_with_key=False, test_with_tf=False, testName='blurry')

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