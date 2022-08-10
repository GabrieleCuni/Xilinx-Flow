import os 
import argparse
import cv2
import preprocess
import numpy as np
import tensorflow as tf

""" ./docker_run.sh xilinx/vitis-ai:1.3.411 """

model = tf.keras.applications.MobileNet(alpha=1, input_shape=(224,224,3))

image = cv2.imread('./calibration_images/ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = preprocess.central_crop(image, 224, 224)
image = preprocess.tf_imagenet_preprocess(image)
image = np.expand_dims(image, 0)
preds = np.array(model(image))
result = tf.keras.applications.mobilenet.decode_predictions(preds, top=5)
# print(result)
print(preds.shape)
print(np.argmax(preds, axis=1))

# model = tf.keras.applications.mobilenet_v2.MobileNetV2()
# model = tf.keras.applications.MobileNet(alpha=1, input_shape=(224,224,3))

# imagePath = os.path.join("calibration_images","ILSVRC2012_img_val","ILSVRC2012_val_00000001.JPEG")   
# image = tf.io.read_file(imagePath)
# image = tf.io.decode_jpeg(image, channels=3)
# image = tf.cast(image, tf.float32)
# image = tf.keras.applications.mobilenet.preprocess_input(image)
# image = tf.expand_dims(image, 0)

# preds = model(image)
# result = tf.keras.applications.mobilenet.decode_predictions(preds, top=5)
# print(result)

# print(model.summary())

# model.save("./tf_models/my_mobilenet")


# model = tf.keras.applications.MobileNet()

# model.save("./tf_models/my_mobilenet")