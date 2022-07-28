import numpy as np
import cv2
import os
import argparse
import time
import preprocess
import sys
import random
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize

def load_and_preprocess(imageSize): # Creare il dataset con anche le label, usa la classe Dataset già per leggere i file 
    images = os.listdir(os.path.join("calibration_images","ILSVRC2012_img_val"))
    random.shuffle(images)
    images = images[:1000]

    preprocessed_images = []
    print("\nStart preprocessing the calibration images\n")
    for i in images:
        img = cv2.imread(os.path.join("calibration_images","ILSVRC2012_img_val",i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess.central_crop(img, imageSize, imageSize)
        img = preprocess.tf_imagenet_preprocess(img)
        # img = np.expand_dims(img, 0)
        if img.shape == (224, 224, 3):
            preprocessed_images.append(img)
    print("\nEnd preprocessing the calibration images\n")

    # return np.array(preprocessed_images)
    return tf.data.Dataset.from_tensor_slices(preprocessed_images)

random.seed(42)

calibrationDataset = load_and_preprocess(224)

batched_dataset = calibrationDataset.batch(32, drop_remainder=True)

with vitis_quantize.quantize_scope():
    quantized_model = tf.keras.models.load_model(os.path.join("tf2_vai_quant_models","my_mobilenet",'quantized_my_mobilenet.h5'))

quantized_model.compile(	
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
	metrics= tf.keras.metrics.SparseTopKCategoricalAccuracy())
quantized_model.evaluate(batched_dataset)