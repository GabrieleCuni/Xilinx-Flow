import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from scipy.stats import t
from pynq_dpu import DpuOverlay
import argparse

def calculate_softmax(data):
    result = np.exp(data)
    return result

def resize_shortest_edge(image, size):
    H, W = image.shape[:2]
    if H >= W:
        nW = size
        nH = int(float(H)/W * size)
    else:
        nH = size
        nW = int(float(W)/H * size)
    return cv2.resize(image,(nW,nH))

def central_crop(image, crop_height, crop_width):
    image_height = image.shape[0]
    image_width = image.shape[1]
    offset_height = (image_height - crop_height) // 2
    offset_width = (image_width - crop_width) // 2
    return image[offset_height:offset_height + crop_height, offset_width:
                offset_width + crop_width, :]

def tf_imagenet_preprocess(x):
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(float, copy=False)

    x /= 127.5
    x -= 1.
    return x

def preprocess_fn(image, size=224, crop_height=224, crop_width=224):
    image = resize_shortest_edge(image, size)
    image = central_crop(image, crop_height, crop_width)
    image = tf_imagenet_preprocess(image)
    return image

def predict_label(softmax):
    with open("img/words.txt", "r") as f:
        lines = f.readlines()
    return lines[np.argmax(softmax)-1]

def confidenceInterval(x, cl=0.99):
    t_sh = t.ppf((cl + 1) / 2, df=len(x) - 1)  # threshold for t_student
    x_hat = x.mean()  
    s = x.std(ddof=1)  # Squared root of the estimated variance. If ddof=0 Biased estimator, if ddof=1 Unbiased estimator 
    delta = t_sh * s / np.sqrt(len(x))  # confidence interval half width
    rel_err = delta / x_hat 
    accuracy = 1 - rel_err
    lowerBound = x_hat - delta
    upperBound = x_hat + delta
    return x_hat, s, lowerBound, upperBound, rel_err, accuracy

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="mobilenet_v1_1.0_224")
parser.add_argument("-d", "--dpu", type=str, default="B4096")
parser.add_argument("-r", "--round", type=int, default=6)
args = parser.parse_args()

imageSize = int(args.model.split("_")[3])

print("*************************")
print("INPUT PARAMETERS:")
print(f"\tModel: {args.model}")
print(f"\tDPU: {args.dpu}")
print(f"\tImage crop size: {imageSize}")
print("*************************")

overlay = DpuOverlay(f"./DPUs/{args.dpu}/dpu.bit")
overlay.load_model(f"./mobilenets/{args.dpu}/{args.model}_{args.dpu}.xmodel")

dpu = overlay.runner

inputTensors = dpu.get_input_tensors()
outputTensors = dpu.get_output_tensors()

shapeIn = tuple(inputTensors[0].dims) # [1,224,224,3]
shapeOut = tuple(outputTensors[0].dims) # [1,1001]
outputSize = int(outputTensors[0].get_data_size() / shapeIn[0]) # 1001
softmax = np.empty(outputSize) # len = 1001

print("*************************************")
print(f"Input Tensor Name: {inputTensors[0].name}")
print(f"Ouput Tensor Name: {outputTensors[0].name}")
print(f"Input Tensor Shape: {shapeIn} ")
print(f"Output Tensor Shape: {shapeOut} ")
print(f"Get Data Size: {outputTensors[0].get_data_size()}")
print(f"outputSize: {outputSize}")
print("*************************************")

output_data = [np.empty(shapeOut, dtype=np.float32, order="C")] # [1,1001]
input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
image = input_data[0]

image_folder = 'img'
original_images = [i for i in os.listdir(image_folder) if i.endswith("JPEG")]
total_images = len(original_images)
print(f"\tNumber of test images: {total_images}")

def run(image_index, display=False, compute=False):
    # strat time
    t0 = time.time()
    picture = cv2.imread(os.path.join(image_folder, original_images[image_index]))
    image[0,...] = preprocess_fn(picture, imageSize, imageSize, imageSize)
    t1= time.time()
    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)
    # print("output_data: ", output_data[0].shape) # list of np array with shape [1,1001]
    temp = [j.reshape(1, outputSize) for j in output_data]
    # print(f"Temp shape: {temp[0].shape} temp len: {len(temp)} temp[0] len: {len(temp[0])} temp[0][0] len: {len(temp[0][0])}")
    softmax = calculate_softmax(temp[0][0])
    # softmax = temp[0][0] # OSS se elimino il softmax la rete da le stesse labels alle 11 immagini
    # print(f"Softmax sum: {sum(softmax)}")
    t2 = time.time()
    # stop time
    if display:
        print(f"Image: {original_images[image_index]} classified as: {predict_label(softmax)}")
       
    if compute:
        classification = predict_label(softmax)
        preprocessing_time = t1-t0
        inference_time = t2-t1
        total_time = t2-t0
        return classification,preprocessing_time,inference_time,total_time

# Con il file arch.json giusto funziona!!! :)
experiment_result = {"c":[],"p":[],"i":[],"t":[]}
for i in range(total_images):
    classification,preprocessing_time,inference_time,total_time = run(i, display=False, compute=True)
    experiment_result["c"].append(classification)
    experiment_result["p"].append(preprocessing_time)
    experiment_result["i"].append(inference_time)
    experiment_result["t"].append(total_time)

rounding = args.round
    
x_hat, s, lowerBound, upperBound, rel_err, accuracy = confidenceInterval(np.array(experiment_result["p"]))
print("Mean preprocessing time: ", round(x_hat,rounding))
x_hat, s, lowerBound, upperBound, rel_err, accuracy = confidenceInterval(np.array(experiment_result["i"]))
print("Mean inference time: ", round(x_hat,rounding))
x_hat, s, lowerBound, upperBound, rel_err, accuracy = confidenceInterval(np.array(experiment_result["t"]))
print("Mean total time: ", round(x_hat,rounding))

del overlay
del dpu
