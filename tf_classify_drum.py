"""
* Read images
* Extract features from each image
* Train a NN using TF with these labeled features
* Test on unknown data
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import tensor_sample_collector
import random

"""
    Admin Functions
"""
# Read from our file
def split_to_test_train(data, train_perc):
    train_set, test_set = [], []
    for d in data:
        if random.randint(0, 101) <= train_perc:
            train_set.append(d)
        else:
            test_set.append(d)
    
    return train_set, test_set

def read_images():
    stick_images, nostick_images = [], []
    for i in range(1, 2093):
        file_name = f'stick_sample/sample_stick_sample{i}.jpg'
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        stick_images.append(img)

        file_name = f'nostick_sample/sample_nostick_sample{i}.jpg'
        img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        nostick_images.append(img)
    
    stick_train, stick_test = split_to_test_train(stick_images, 80)
    nostick_train, nostick_test = split_to_test_train(nostick_images, 80)

    print(f'Obtained {len(stick_train)} stick training set.')
    print(f'Obtained {len(nostick_train)} nostick training set.')
    print(f'Obtained {len(stick_test)} stick testing set.')
    print(f'Obtained {len(nostick_test)} nostick testing set.')

    return (nostick_train, stick_train, nostick_test, stick_test)

# Dictionary to map each digit to its list of images
def make_digit_map(data):
    digit_map = {i:[] for i in range(2)}
    for digit in range(2):
        for img in data[digit]:
            digit_map[digit].append(img)
    
    return digit_map

# Extract features
# fnlist is a list of feature-generating functions, each of which should
#   take a 28x28 grayscale (0-255) image, 0=white, and return a 1-d array
#   of numbers
# Returns a map: digit -> nparray of feature vectors, one row per image
def build_feature_map(digit_map, fnlist):
    fmap = {i:[] for i in range(2)}
    count = 0
    for digit in fmap:
        for img in digit_map[digit]:
            feature_vector = []
            for f in fnlist:
                feature_vector += f(img)
            fmap[digit].append(feature_vector)
    return fmap

"""
    TESTS
"""
# Tests a feature on the traninig data.
def test_nn(feature):
    print('Building feature map')
    feature_map = build_feature_map(test_digit_map, [feature])
    print('Done building feature map')

    print('Started training')
    nn = get_trained_nn(feature)
    print('Obtained Neural Network')

    #preds = nn.predict_classes(feature_map)
    #print(preds)
    
    success = 0.0
    failure = 0.0
    predictions = [[0 for i in range(2)] for j in range(2)]

    total_len = 0
    for f in feature_map.keys():
        total_len += len(feature_map[f])
    
    perc10 = total_len//10
    count = 0

    for i in range(2):
        print(f'Len of {i}s feature map is {len(feature_map[i])}')

    for digit in range(2):
        for f in feature_map[digit]:
            unkn = np.array([f]).astype(np.float32)
            prediction = nn.predict_classes(unkn)[0]
            predictions[digit][int(prediction)] += 1
            count += 1
            if prediction == digit:
                success += 1
            else:
                failure += 1
            if count % perc10 == 0:
                print(f"Tested {count}/{total_len}")
            
    print(success/(success + failure))
    print(np.array(predictions))


# Returns a trained KNN object for a feature.
def get_trained_nn(feature):
    print("\n\n", feature.__name__)

    # train
    feature_map = build_feature_map(training_digit_map, [feature])

    train = []
    labels = []
    for digit in range(2):
        for f in feature_map[digit]:
            train.append(f)
            labels.append(digit)
        
    print('Creating model ...')

    train = np.asarray(train)
    labels = np.asarray(labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1000, activation='sigmoid', name='fc1', input_shape=(len(train[0]),)),
        tf.keras.layers.Dense(10, name='fc2', activation='softmax')])
    print('Done creating model ... now compiling')

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('Done compiling model ... now fitting')

    model.fit(train, labels, epochs=30, batch_size=10, verbose=1)
    model.save("models/first_drumstick_model")
    print('Done fitting model!')
    print('COMPLETE!')

    return model

# The "if" statement below means to run the code if below, only if this
# is the Python file being run, for example, if the user typed
# "python tf_classify.py" on the command line.
# But if someone imports this it will not be run.
if __name__ == "__main__":
    print('Started reading training and test data')
    main_data = read_images()
    training_data = (main_data[0], main_data[1])
    test_data = (main_data[2], main_data[3])

    training_digit_map = make_digit_map(training_data)
    test_digit_map = make_digit_map(test_data)

    print("Training digit map>>>")
    print(training_digit_map)
    print("Test digit map>>>")
    print(test_digit_map)

    all_features = [tensor_sample_collector.weighted_sectional_density]
    #all_features = [tensor_sample_collector.darkness_gradient]

    for f in all_features:
        test_nn(f)