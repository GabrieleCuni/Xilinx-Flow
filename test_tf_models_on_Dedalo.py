import argparse
import random
import os
import cv2
import preprocess
import numpy as np
import tensorflow as tf
# import tensorflow.contrib.decent_q

def test_calibration(x_test, y_test, y_label):
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    with tf.io.gfile.GFile(f"./tf_models/{args.model}/{args.model}_frozen.pb", "rb") as f:
        graph = tf.compat.v1.GraphDef()
        graph.ParseFromString(f.read()) # This upload the model graph in RAM

    tf.import_graph_def(graph,name = '') # This is the model load operation

    # input data Tensor("input:0", shape=(?, 224, 224, 3), dtype=float32)
    input_data = tf.compat.v1.get_default_graph().get_tensor_by_name('input'+':0')
    # labels Tensor("Placeholder:0", shape=(?,), dtype=int64)
    labels = tf.compat.v1.placeholder(tf.int64, shape=[None,])
    # logits shape [?,1001] dtype=float32
    logits = tf.compat.v1.get_default_graph().get_tensor_by_name('MobilenetV1/Predictions/Reshape_1'+':0')

    # nn_output shape [?,] dtype=int64
    nn_output = tf.argmax(logits, 1)
    # correct prediction Tensor("Equal:0", shape=(?,), dtype=bool)
    correct_prediction = tf.equal(nn_output, labels)
    # accuracy Tensor("Mean:0", shape=(), dtype=float32)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # accuracy metric
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer()) # An Op that initializes global variables in the graph. This is just a shortcut for variables_initializer(global_variables())
        sess.run(tf.compat.v1.initializers.local_variables()) # An Op that initializes all local variables in the graph. This is just a shortcut for variables_initializer(local_variables())
        print("Start model evaluation")
        # t.eval() is a shortcut for calling tf.get_default_session().run(t)
        # tf.tensor.eval(feed_dict=None, session=None)
        # feed_dict := A dictionary that maps Tensor objects to feed values. See tf.Session.run for a description of the valid feed values. 
        acc = accuracy.eval(feed_dict={input_data: x_test, labels: y_test})
        print("Average accuracy on validation set: {}".format(acc))
        print("Stop model evaluation")

def my_test_calibration(x_test, y_test, y_label):
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    with tf.io.gfile.GFile(f"./tf_models/{args.model}/{args.model}_frozen.pb", "rb") as f: 
        graph = tf.compat.v1.GraphDef()
        graph.ParseFromString(f.read()) # This upload the model graph in RAM

    tf.import_graph_def(graph,name = '') # This is the model load operation

    # input data Tensor("input:0", shape=(?, 224, 224, 3), dtype=float32)
    input_data = tf.compat.v1.get_default_graph().get_tensor_by_name('input'+':0')
    # labels Tensor("Placeholder:0", shape=(?,), dtype=int64)
    # labels = tf.compat.v1.placeholder(tf.int64, shape=[None,])
    # logits shape [?,1001] dtype=float32
    logits = tf.compat.v1.get_default_graph().get_tensor_by_name('MobilenetV1/Predictions/Reshape_1'+':0')

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
        for img, truth in zip(x_test,y_test):
            predict = nn_output.eval(feed_dict={input_data:img})
            predList.append(predict[0])
            if int(predict[0]) == int(truth):
                count += 1
        
        accuracy = count / len(y_test)
        print(f"Accuracy: {accuracy}")
        print(f"Correct inference: {count}")
        print(f"Total inference: {len(y_test)}\n")
        for x,y,z in zip(predList, y_test, y_label):
            print(f"Pred: {x}, GroundTruth: {y}, Label: {z}")

def makeTestSet(imageSize):
    x_test = []
    y_test = []
    y_label = []
    valDir = "n15075141" #"n01440764" # "n15075141"
    TestImagePath = os.path.join("/","space_fast","imagenet_folder","val",valDir)
    GroudTruthFile = os.path.join("map_clsloc.txt")

    image_names = os.listdir(TestImagePath)

    with open(GroudTruthFile, "r") as f:
        fileLinesList = f.readlines()
        mapping = {}
        for line in fileLinesList:
            lineSplit = line.split(" ")
            dedaloId = lineSplit[0]
            imagenetId = lineSplit[1]
            Truelabel = lineSplit[2]
            mapping[dedaloId] = (imagenetId,Truelabel)

    print("Start preprocessing the test images")
    for imgName in image_names:
        img = cv2.imread(os.path.join(TestImagePath, imgName))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = preprocess.resize_shortest_edge(img, imageSize)
        # img = preprocess.central_crop(img, imageSize, imageSize)
        # img = preprocess.tf_imagenet_preprocess(img)
        img = preprocess.finalPreprocess(image=img, height=imageSize, width=imageSize)
        img = np.expand_dims(img, axis=0)
        imagenetId, Truelabel = mapping[valDir]
        # print(f"Image name: {imgName}, groundTruth: {imagenetId}, label: {Truelabel}")
        y_label.append(Truelabel)
        y_test.append(imagenetId)
        x_test.append(img)
    print("Stop preprocessing the test images\n")

    return x_test, y_test, y_label

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="mobilenet_v1_1.0_224")
parser.add_argument("-n", "--number", type=int, default=30)
args = parser.parse_args()

random.seed(42)
x_test, y_test, y_label = makeTestSet(imageSize=int(args.model.split("_")[3]))
# print(x_test[0].shape, y_test[0])
my_test_calibration(x_test, y_test, y_label)
