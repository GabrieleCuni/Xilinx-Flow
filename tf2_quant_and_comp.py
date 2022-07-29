import numpy as np
import cv2
import os
import argparse
import time
import preprocess
import sys
import random
from myUtils import DatasetGenerator
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize

datasetGenerator = DatasetGenerator(batch_size=32, imageNumber=100, width=224, height=224)
batched_dataset = datasetGenerator.make_dataset()


modelPath = os.path.join("tf_models","my_mobilenet")
float_model = tf.keras.models.load_model(modelPath, compile=False)

quantizer = vitis_quantize.VitisQuantizer(float_model)
quantized_model = quantizer.quantize_model(calib_dataset=batched_dataset) 

quantized_model.save(os.path.join("tf2_vai_quant_models","my_mobilenet","quantized_my_mobilenet.h5"))