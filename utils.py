
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import consts

def get_saved_model(model_name):
    return tf.keras.models.load_model(model_name)

def preprocess_for_tf(img):
    bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(bw_img, (consts.IMG_DIM, consts.IMG_DIM))
        
def predict(img, probability_model, preprocess=False, reshape_param=[80,80,1]):
    if preprocess:
        img = preprocess_for_tf(img)
    if reshape_param is not None:
        img = tf.reshape(img, reshape_param)
    unkn = np.array([np.asarray(img)]).astype(np.float32)
    predict_classes = probability_model.predict_classes(unkn)
    print(predict_classes)
    return predict_classes[0], img
