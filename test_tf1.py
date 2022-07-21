import argparse
import random
import os
import cv2
import preprocess
import numpy as np
import tensorflow as tf
import tensorflow.contrib.decent_q

GroudTruthFile = os.path.join("calibration_images","ILSVRC2012_validation_ground_truth.txt")

with open(GroudTruthFile, "r") as f:
        mapping = f.readlines()

def getImageNumber(image_name):
    return int(image_name.split("_")[2].split(".")[0])

# the image is a Snake on a beach/sand field
img = cv2.imread(os.path.join("./calibration_images/ILSVRC2012_img_val","ILSVRC2012_val_00000001.JPEG"))
img = preprocess.resize_shortest_edge(img, 224)
img = preprocess.central_crop(img, 224, 224)
img = preprocess.tf_imagenet_preprocess(img)
img = np.expand_dims(img, axis=0)
groundTruth  = mapping[getImageNumber("ILSVRC2012_val_00000001.JPEG")-1]

tf.compat.v1.reset_default_graph()
with tf.io.gfile.GFile(f"./tf_models/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_frozen.pb", "rb") as f:
    graph = tf.compat.v1.GraphDef()
    graph.ParseFromString(f.read()) # This upload the model graph in RAM

tf.import_graph_def(graph,name = '') # This is the model load operation

input_data = tf.compat.v1.get_default_graph().get_tensor_by_name('input'+':0')
labels = tf.compat.v1.placeholder(tf.int64, shape=[None,])
logits = tf.compat.v1.get_default_graph().get_tensor_by_name('MobilenetV1/Predictions/Reshape_1'+':0')

nn_output = tf.argmax(logits, 1)
# correct_prediction = tf.equal(nn_output, labels)
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # accuracy metric

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer()) # An Op that initializes global variables in the graph. This is just a shortcut for variables_initializer(global_variables())
    sess.run(tf.compat.v1.initializers.local_variables()) # An Op that initializes all local variables in the graph. This is just a shortcut for variables_initializer(local_variables())
    # t.eval() is a shortcut for calling tf.get_default_session().run(t)
    # tf.tensor.eval(feed_dict=None, session=None)
    # feed_dict := A dictionary that maps Tensor objects to feed values. See tf.Session.run for a description of the valid feed values. 
    # acc = accuracy.eval(feed_dict={input_data: img, labels: y_test})
    # print("Average accuracy on test set: {}".format(acc))
    predict = nn_output.eval(feed_dict={input_data:img})
    print()
    print("Predict",predict, "GroudTruth", groundTruth)
    print("Predict",type(predict), "GroudTruth", type(groundTruth))
