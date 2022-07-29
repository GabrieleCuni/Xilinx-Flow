import os 
import argparse
import cv2
import preprocess
import numpy as np
import tensorflow as tf

""" ./docker_run.sh xilinx/vitis-ai:1.3.411 """

model = tf.keras.applications.MobileNet()

image = cv2.imread('./calibration_images/ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = preprocess.central_crop(image, 224, 224)
image = preprocess.tf_imagenet_preprocess(image)
image = np.expand_dims(image, 0)
preds = np.array(model(image))
result = tf.keras.applications.mobilenet.decode_predictions(preds, top=5)
print(result)
print(preds.shape)
print(np.argmax(preds, axis=1))

# # print(model.summary())

# model.save("./tf_models/my_mobilenet")


# model = tf.keras.applications.MobileNet()

# model.save("./tf_models/my_mobilenet")