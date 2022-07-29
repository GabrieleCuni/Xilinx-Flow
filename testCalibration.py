import argparse
import random
import os
import cv2
import preprocess
import json
import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.decent_q

"""
jupyter notebook --ip=0.0.0.0 --port=8080
"""

def my_test_calibration(x_test, y_test, y_label):
    with open(os.path.join("calibration_images","imagenet_class_index.json")) as fp:
        classIndex = json.load(fp)

    tf.compat.v1.reset_default_graph()
    # f"./tf_models/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224_frozen.pb"
    if args.quantize:
         with tf.io.gfile.GFile(f"./vai_quant_models/{args.model}/quantize_eval_model.pb", "rb") as f: # f"./vai_quant_models/{args.model}/quantize_eval_model.pb" 
            graph = tf.compat.v1.GraphDef()
            graph.ParseFromString(f.read()) # This upload the model graph in RAM
    else:
        with tf.io.gfile.GFile(f"./tf_models/{args.model}/{args.model}_frozen.pb", "rb") as f: # f"./vai_quant_models/{args.model}/quantize_eval_model.pb" 
            graph = tf.compat.v1.GraphDef()
            graph.ParseFromString(f.read()) # This upload the model graph in RAM

    tf.import_graph_def(graph,name = '') # This is the model load operation

    # input data Tensor("input:0", shape=(?, 224, 224, 3), dtype=float32)
    input_data = tf.compat.v1.get_default_graph().get_tensor_by_name('input'+':0')
    # labels Tensor("Placeholder:0", shape=(?,), dtype=int64)
    # labels = tf.compat.v1.placeholder(tf.int64, shape=[None,])
    # logits shape [?,1001] dtype=float32
    logits = tf.compat.v1.get_default_graph().get_tensor_by_name(args.outputLayer +':0') # Se metto reshape o reshape_1 l'accuratezza rimane la stessa

    # nn_output shape [?,] dtype=int64
    nn_output = tf.argmax(logits, 1)
    # correct prediction Tensor("Equal:0", shape=(?,), dtype=bool)
    # correct_prediction = tf.equal(nn_output, labels)
    # accuracy Tensor("Mean:0", shape=(), dtype=float32)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # accuracy metric
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer()) # An Op that initializes global variables in the graph. This is just a shortcut for variables_initializer(global_variables())
        sess.run(tf.compat.v1.initializers.local_variables()) # An Op that initializes all local variables in the graph. This is just a shortcut for variables_initializer(local_variables())
        print()
        print("Start Inference")
        # t.eval() is a shortcut for calling tf.get_default_session().run(t)
        # tf.tensor.eval(feed_dict=None, session=None)
        # feed_dict := A dictionary that maps Tensor objects to feed values. See tf.Session.run for a description of the valid feed values. 

        # acc = accuracy.eval(feed_dict={input_data: x_test, labels: y_test})
        count = 0
        predList = []
        for img, imagenetId in zip(x_test,y_test):
            predict = nn_output.eval(feed_dict={input_data:img})
            predList.append(predict[0])
            [predId, predLabel] = classIndex[str(predict[0]-1)] # non so se va messo -1
            if predId == imagenetId:
                count += 1
        
        accuracy = count / len(y_test)
        print(f"Accuracy: {accuracy}")
        print(f"Correct inference: {count}")
        print(f"Total inference: {len(y_test)}\n")
        
def getImageNumber(image_name):
    return int(image_name.split("_")[2].split(".")[0])

def makeTestSet(imageSize): # controlla che tutte le foto abbiano la stessa dimensione: Secondo me si perché facciamo cv2.resize
    x_test = []
    y_test = []
    y_label = []
    TestImagePath = os.path.join("calibration_images","ILSVRC2012_img_val")
    GroudTruthFile = os.path.join("calibration_images","ILSVRC2012_validation_ground_truth.txt")
    mapClsloc = os.path.join("calibration_images","map_clsloc.txt")

    image_names = os.listdir(TestImagePath)
    random.shuffle(image_names)
    image_names = image_names[1024:2024]

    with open(GroudTruthFile, "r") as f:
        mapping = f.readlines()

    with open(mapClsloc, "r") as fp:
        mapClslocMapping = fp.readlines()

    print("Start preprocessing the test images")
    for i in image_names:
        img = cv2.imread(os.path.join("./calibration_images/ILSVRC2012_img_val",i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = preprocess.resize_shortest_edge(img, imageSize)
        # img = preprocess.central_crop(img, imageSize, imageSize)
        # img = preprocess.tf_imagenet_preprocess(img)
        img = preprocess.finalPreprocess(image=img, height=imageSize, width=imageSize)
        img = np.expand_dims(img, axis=0)
        groundTruth  = mapping[getImageNumber(i)-1]
        stringToSplit = mapClslocMapping[int(groundTruth)-1]
        imagenetID = stringToSplit.split(" ")[0]
        label = stringToSplit.split(" ")[2]
        y_test.append(imagenetID)
        y_label.append(label)
        x_test.append(img)
    print("Stop preprocessing the test images\n")

    return x_test, y_test, y_label

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="mobilenet_v1_1.0_224")
parser.add_argument("-q", "--quantize", action='store_true')
parser.add_argument("-o", "--outputLayer", type=str, default='MobilenetV1/Predictions/Reshape_1')

args = parser.parse_args()

print("************************************")
print("INPUT PARAMETERS:")
print(f"\tmodel: {args.model}")
print(f"\tOutput Layer: {args.outputLayer}")
print(f"\tValidate qantized model: {args.quantize}")
print(f"\tValidate original tensorflow model: {not args.quantize}")
print("************************************")

random.seed(42)
x_test, y_test, y_label = makeTestSet(imageSize=int(args.model.split("_")[3]))
# print(x_test[0].shape, y_test[0])
my_test_calibration(x_test, y_test, y_label)





















# def test_calibration(x_test, y_test):
#     tf.reset_default_graph()
#     with tf.gfile.GFile(f"./vai_quant_models/{args.model}/quantize_eval_model.pb", "rb") as f:
#         graph = tf.GraphDef()
#         graph.ParseFromString(f.read()) # This upload the model graph in RAM

#     tf.import_graph_def(graph,name = '') # This is the model load operation

#     # input data Tensor("input:0", shape=(?, 224, 224, 3), dtype=float32)
#     input_data = tf.get_default_graph().get_tensor_by_name('input'+':0')
#     # labels Tensor("Placeholder:0", shape=(?,), dtype=int64)
#     labels = tf.placeholder(tf.int64, shape=[None,])
#     # logits shape [?,1001] dtype=float32
#     logits = tf.get_default_graph().get_tensor_by_name('MobilenetV1/Predictions/Reshape_1'+':0')

#     # nn_output shape [?,] dtype=int64
#     nn_output = tf.argmax(logits, 1)
#     # correct prediction Tensor("Equal:0", shape=(?,), dtype=bool)
#     correct_prediction = tf.equal(nn_output, labels)
#     # accuracy Tensor("Mean:0", shape=(), dtype=float32)
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # accuracy metric
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer()) # An Op that initializes global variables in the graph. This is just a shortcut for variables_initializer(global_variables())
#         sess.run(tf.initializers.local_variables()) # An Op that initializes all local variables in the graph. This is just a shortcut for variables_initializer(local_variables())
#         print("Start model evaluation")
#         # t.eval() is a shortcut for calling tf.get_default_session().run(t)
#         # tf.tensor.eval(feed_dict=None, session=None)
#         # feed_dict := A dictionary that maps Tensor objects to feed values. See tf.Session.run for a description of the valid feed values. 
#         acc = accuracy.eval(feed_dict={input_data: x_test, labels: y_test})
#         print("Average accuracy on test set: {}".format(acc))
#         print("Stop model evaluation")


























# path = "/space_fast/imagenet_folder/val/"
# image_dir = os.listdir(path)

# x_test = []
# y_test = []

# for dir in image_dir:
#     image_list = os.listdir(os.path.join(path, dir))
#     random.shuffle(image_list)
#     image_name = image_list[0]





