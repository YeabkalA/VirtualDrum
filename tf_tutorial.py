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
import threading
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

RESIZE_DIM = 80
GLOBAL_ADAPTIVE_THRESHOLD = False
CONVOLUTIONAL = False

def create_multiple_images(img, adaptive_threshold=0):
    img = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM))
    images = []

    for rotate_factor in [cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, -190]:
        if rotate_factor == -190:
            img2 = img.copy()
        else:
            img2 = cv2.rotate(img, rotate_factor)
        images.append(img2)

    return images
# def create_multiple_images(img, adaptive_threshold=GLOBAL_ADAPTIVE_THRESHOLD):
#     images = []

#     for flip_factor in [-1, 1]: # Flip horizontally or no flip.
#         if flip_factor>0:
#             img1 = cv2.flip(img, flip_factor)
#         else:
#             img1 = img.copy()

#         for rotate_factor in [cv2.ROTATE_180, -190]:
#             if rotate_factor == -190:
#                 img2 = img1.copy()
#             else:
#                 img2 = cv2.rotate(img1, rotate_factor)
#             for contrast in [1]:
#                 img3 = tf.image.adjust_contrast(img2, contrast_factor=contrast)
#                 for bright in [0.0]:
#                     img4 = tf.image.adjust_brightness(img3, delta=bright)
#                     img4 = img4.numpy()
#                     img_resized = cv2.resize(img4, (RESIZE_DIM, RESIZE_DIM))
#                     if CONVOLUTIONAL:
#                         img_resized = tf.reshape(img_resized, [RESIZE_DIM, RESIZE_DIM, 1])

#                     if adaptive_threshold:
#                         _, img_resized = cv2.threshold(img_resized,105,255,cv2.THRESH_BINARY)

#                     images.append(img_resized)

#     return images
    #return [cv2.resize(img, (RESIZE_DIM, RESIZE_DIM))]

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

def test_blurry(adaptive_threshold=GLOBAL_ADAPTIVE_THRESHOLD):
    rv = []

    num_test_stick_samples = get_num_blurry_stick_samples()
    for i in range(1, num_test_stick_samples + 1):
        file_name = get_file_name(tensor_sample_collector.BLURRY_DATATYPE, i)
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM))
        
        if adaptive_threshold:
            _,img_resized = cv2.threshold(img_resized,105,255,cv2.THRESH_BINARY)

        if CONVOLUTIONAL:
            img_resized = tf.reshape(img, [RESIZE_DIM, RESIZE_DIM, 1])
        rv.append((np.asarray(img_resized), 0))
    
    images, labels = [],[]
    for data in rv:
        images.append(data[0])
        labels.append(data[1])
    
    return np.asarray(images), np.asarray(labels)


def test_specific_images(adaptive_threshold=GLOBAL_ADAPTIVE_THRESHOLD):
    rv = []

    num_test_stick_samples = get_num_test_stick_samples()
    for i in range(1, num_test_stick_samples + 1):
        file_name = get_file_name(tensor_sample_collector.TEST_STICK_DATATYPE, i)
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM))

        if adaptive_threshold:
            _, img_resized = cv2.threshold(img_resized,105,255,cv2.THRESH_BINARY)
        
        if CONVOLUTIONAL:
            img_resized = tf.reshape(img, [RESIZE_DIM, RESIZE_DIM, 1])
        rv.append((np.asarray(img_resized), 0))
    
    num_test_no_stick_samples = get_num_test_no_stick_samples()
    for i in range(1, num_test_no_stick_samples + 1):
        file_name = get_file_name(tensor_sample_collector.TEST_NO_STICK_DATATYPE, i)
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (RESIZE_DIM, RESIZE_DIM))
        rv.append((np.asarray(img_resized), 1))

    random.shuffle(rv)

    images, labels = [],[]
    for data in rv:
        images.append(data[0])
        labels.append(data[1])
    
    return np.asarray(images), np.asarray(labels)

def read_directory(dir_name, label, adaptive_threshold=GLOBAL_ADAPTIVE_THRESHOLD):
    num_samples = get_last_file_name(f'{tensor_sample_collector.ROOT_SAMPLES_DIR}/{dir_name}/')
    images = []
    perc10 = num_samples // 10

    for i in range(1, num_samples + 1):
        if i % perc10 == 0:
            read_percentage = round(i*100/num_samples, 3)
            print(f'Finished reading {read_percentage}% of {dir_name}')
        file_name = get_file_name(dir_name, i)
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        #img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        if len(img) == 0: # Image has been deleted
            continue
        expanded_images_set = create_multiple_images(img, adaptive_threshold)
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

def read_images_modularized(directory_data, adaptive_threshold=GLOBAL_ADAPTIVE_THRESHOLD):
    train_images, train_labels, test_images, test_labels = [], [], [], []

    for dir_info in directory_data:
        dir_name, dir_label = dir_info
        dir_train_data, dir_test_data, dir_train_labels, dir_test_labels = \
            read_directory(dir_name, dir_label, adaptive_threshold)
        train_images += dir_train_data
        train_labels += dir_train_labels
        test_images += dir_test_data
        test_labels += dir_test_labels
    
    #Count
    counts = [0 for i in range(len(directory_data))]
    for label in train_labels:
        counts[label] += 1
    
    print('COUNT SUMMARY--------------')
    print(f'Training on {counts[0]} stick data.')
    print(f'Training on {counts[1]} no stick data.')
    
    return np.asarray(train_images), np.asarray(train_labels), np.asarray(test_images), np.asarray(test_labels)


def read_images(adaptive_threshold=GLOBAL_ADAPTIVE_THRESHOLD):
    directory_data = []
    directory_data.append((tensor_sample_collector.NO_STICK_WHITENESS_ADDED, 1))
    directory_data.append((tensor_sample_collector.STICK_DATATYPE, 0))
    directory_data.append((tensor_sample_collector.NO_STICK_DATATYPE, 1))
    directory_data.append((tensor_sample_collector.BLURRY_DATATYPE, 0))
    # directory_data.append((tensor_sample_collector.NO_STICK_DATATYPE_RESERVE, 1))

    # # Reps!
    # directory_data.append((tensor_sample_collector.NO_STICK_DATATYPE, 1))
    
    
    # directory_data.append((tensor_sample_collector.STICK_WITH_BACKGROUND_DATATYPE, 3))
    # directory_data.append((tensor_sample_collector.NO_STICK_BUT_BACKGROUND_DATATYPE, 4))
    # #directory_data.append((tensor_sample_collector.EDGE_STICK_DATATYPE, 0))
    # #directory_data.append((tensor_sample_collector.FAR_STICK_IMAGES, 0))

    return read_images_modularized(directory_data, adaptive_threshold)

def create_nn(adaptive_threshold=GLOBAL_ADAPTIVE_THRESHOLD):
    print('Started nn creation')
    data = read_images(adaptive_threshold)
    train_images, train_labels, test_images, test_labels = data
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    print('read images...defining model...')
    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(RESIZE_DIM, RESIZE_DIM, 3)),
    #     keras.layers.Dense(128, activation='relu'),
    #     keras.layers.Dense(5)
    # ])
    
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(RESIZE_DIM, RESIZE_DIM)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2)
    ])
    print('defined model!')

    # # Convolutional NN
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Conv2D(64, (3,3), activation = 'relu',
    #         input_shape=(RESIZE_DIM, RESIZE_DIM, 1)),
    #     tf.keras.layers.MaxPooling2D(2,2),
    #     tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    #     tf.keras.layers.MaxPooling2D(2,2),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(2, activation ='softmax')
    # ])

#     model = Sequential([
#         Conv2D(16, 3, padding='same', activation='relu', input_shape=(RESIZE_DIM, RESIZE_DIM, 1)),
#         MaxPooling2D(),
#         Conv2D(32, 3, padding='same', activation='relu'),
#         MaxPooling2D(),
#         Conv2D(64, 3, padding='same', activation='relu'),
#         MaxPooling2D(),
#         Flatten(),
#         Dense(512, activation='relu'),
#         Dense(2)
# ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
        
    print('model compiled, fit to start...')

    model.fit(train_images, train_labels, epochs=20)
    print('model fit end')
    _, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    model.save('models/80-CV2DIFF')

    return model, data
    
# Use the same path to derive images as used for training set.
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

def get_model_for_testing(train=False, modelName='', adaptive_threshold=GLOBAL_ADAPTIVE_THRESHOLD, test=True):
    if train:
        model, data = create_nn(adaptive_threshold=adaptive_threshold)
        if test:
            test_nn(model, (data[2], data[3]), testName='random test data')
            #test_nn(model, test_specific_images(), testName='test specific data')
            #test_nn(model, test_blurry(), testName='blurry')
        return model
    else:
        model = tf.keras.models.load_model(modelName)
        if test:
            data = read_images(adaptive_threshold)
            test_nn(model, (data[2], data[3]), testName='random test data')
            #test_nn(model, test_specific_images(), testName='test specific data')
            #test_nn(model, test_blurry(), testName='blurry')
        return model

def test_live(model, adaptive_threshold=GLOBAL_ADAPTIVE_THRESHOLD):
    # if True:
    #     return
    cap = cv2.VideoCapture(0)
    # drum_area = DrumArea(top_left_corner=(900, 100), square_dim=RESIZE_DIM, sound='c')
    # drum_area2 = DrumArea(top_left_corner=(100, 100), square_dim=RESIZE_DIM, sound='j')
    # drum_area3 = DrumArea(top_left_corner=(100, 400), square_dim=RESIZE_DIM, sound='k')

    drum_area1 = DrumArea(top_left_corner=(50, 50), square_dim=320, sound='c')
    drum_area2 = DrumArea(top_left_corner=(800, 50), square_dim=80, sound='j')

    dareas = [drum_area1]
    area_listener2 = AreaListener(drum_areas=dareas)
    img_process2 = ImageProcessor()
    probability_model2 = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    base_set = False
    base_imgs = None

    while True:
        _, frame_orig = cap.read()
        frame_orig = img_process2.horizontal_flip(frame_orig)
        frame_color = frame_orig.copy()
        #frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        frame = frame_color.copy()

        if not base_set:
            area_listener2.set_base_image(frame)
            base_imgs = area_listener2.get_base_imgs(resize_dim=(RESIZE_DIM, RESIZE_DIM))
            base_set = True
        
        for drum_area in dareas:
            process_area_cv2Diff(drum_area, frame_orig, probability_model2)
            

        # targeted_areas = area_listener2.get_all_target_areas(img=frame)
        # area_id = 0
        area_listener2.draw_area(frame_color)
        # area_listener2.draw_area(frame_color)
        # for targeted_area in dareas:
        #     # thread= threading.Thread(target=process_area, args=(targeted_area, area_id, frame, frame_color, probability_model2,))
        #     # thread.start()
        #     process_area(targeted_area, area_id, frame, frame_color, probability_model2)
        key = cv2.waitKey(1)
        # if key == ord('s'):
        #     pass

        cv2.imshow('Main', frame_color)
        cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break

def process_area_cv2Diff(drum_area, frame_orig, probability_model):
    orig, target = drum_area.get_area(frame_orig), drum_area.base_img
    diff = cv2.absdiff(target, orig)
    diff_gray = np.asarray(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY))

    file_name = f'tp{drum_area.id}.jpg'
    cv2.imwrite(file_name, diff_gray)
    to_predict_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    # to_predict_img = tf.reshape(to_predict_img, [RESIZE_DIM, RESIZE_DIM, 1])
    to_predict_img = cv2.resize(to_predict_img, (RESIZE_DIM, RESIZE_DIM))

    cv2.imshow(f'Target{drum_area.id}', to_predict_img)
    to_predict_list = np.asarray([np.asarray(to_predict_img)])
    prediction, _ = utils.predict(to_predict_list[0], probability_model, preprocess=False, reshape_param=None)

    if prediction == 0:
        print('Stick')


def process_area(targeted_area, frame, frame_color, probability_model2):
    file_name = f'tp{area_id}.jpg'
    cv2.imwrite(file_name, targeted_area.get_area(frame))
    to_predict_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    if CONVOLUTIONAL:
        to_predict_img = tf.reshape(to_predict_img, [RESIZE_DIM, RESIZE_DIM, 1])
    
    to_predict_img = cv2.resize(to_predict_img, (RESIZE_DIM, RESIZE_DIM))
    cv2.imshow(f'Target{area_id}', to_predict_img)
    to_predict_list = np.asarray([np.asarray(to_predict_img)])
    prediction, _ = utils.predict(to_predict_list[0], probability_model2, preprocess=False, reshape_param=None)

    if prediction == 0:
        cv2.circle(frame_color, (500,100*area_id), 90, ((area_id * 30 + 430) % 255,(area_id * 100) % 255, (20 * area_id) % 255), -1)
        targeted_area.playSound()
        targeted_area.markPlayed(frame_color)
    

def show_thresholds():
    data = read_directory(tensor_sample_collector.BLURRY_DATATYPE, 1)
    train_images = data[0]
    for image in train_images:
        ret,thresh1 = cv2.threshold(image,105,255,cv2.THRESH_BINARY)
        cv2.imshow('Threshold', thresh1)
        cv2.waitKey(0)


# for i in range(1, 2000, 300):
#     img = cv2.imread(get_file_name(tensor_sample_collector.STICK_DATATYPE, i))
#     cv2.imshow('No contrast', img)
#     cv2.waitKey()
#     img2 = tf.image.adjust_contrast(img, contrast_factor=5).numpy()
#     cv2.imshow('contrast', img2)
#     cv2.waitKey()



# show_thresholds()
model = get_model_for_testing(train=False, modelName='models/80-CV2DIFF', test=True)
model.summary()
test_live(model)