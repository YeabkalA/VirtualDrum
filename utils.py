
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
        
def predict(img, probability_model, preprocess=False):
    if preprocess:
        img = preprocess_for_tf(img)
    unkn = np.array([np.asarray(img)]).astype(np.float32)
    return probability_model.predict_classes(unkn)[0], img
