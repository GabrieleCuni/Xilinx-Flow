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

def load_and_preprocess(imageSize):
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
print(f"\ncalibrationDataset len: {len(calibrationDataset)}\n")

batched_dataset = calibrationDataset.batch(32, drop_remainder=True)



modelPath = os.path.join("tf_models","my_mobilenet")
float_model = tf.keras.models.load_model(modelPath, compile=False)

quantizer = vitis_quantize.VitisQuantizer(float_model)
quantized_model = quantizer.quantize_model(calib_dataset=batched_dataset) 

quantized_model.save(os.path.join("tf2_vai_quant_models","my_mobilenet","quantized_my_mobilenet.h5"))