"""
Before running this script the current working directory have to be:
    /home/gabriele/DPU-PYNQ/host
Then execute the command:
    ./docker_run.sh xilinx/vitis-ai:1.3.411
Now you are in the Vitis AI docker system
Then execute the command:
    conda activate vitis-ai-tensorflow
Now you sourced the Vitis AI Tensorflow environment
Thanks to the above command sequence the "vai_q_tensorflow" and "vai_c_tensorflow" command are available
Now you must execute:
    cd ThesisData 
You can run the scrips with:
    python VitisAI_quantize_and_compile_no_zoo_model_flow.py

# Arch.json file Location
I have move this file:
/DPU-PYNQ/boards/ZCU104/binary_container_1/link/vivado/vpl/prj/prj.gen/sources_1/bd/dpu/ip/dpu_DPUCZDX8G_1_0/arch.json

In this location:
/home/gabriele/DPU-PYNQ/host/ThesisData/Arch_files/
"""

import numpy as np
import cv2
import os
import argparse
import time
import preprocess
import sys
import random

def load_and_preprocess(imageSize):
    images = os.listdir(os.path.join("calibration_images","ILSVRC2012_img_val"))
    random.shuffle(images)
    images = images[:1024]

    preprocessed_images = []
    print("Start preprocessing the calibration images")
    for i in images:
        img = cv2.imread(os.path.join("calibration_images", "ILSVRC2012_img_val", i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess.finalPreprocess(image=img, height=imageSize, width=imageSize)
        preprocessed_images.append(img)
    print("End preprocessing the calibration images")

    print(f"Saving {len(preprocessed_images)} images in ./PreprocessCalibImages/calib_data.npz")
    print("...")
    np.savez(os.path.join("PreprocessCalibImages","calib_data.npz"), data = preprocessed_images)
    print("Saving with success on disk")

modelList = [
            "mobilenet_v1_1.0_224", "mobilenet_v1_1.0_192", "mobilenet_v1_1.0_160", "mobilenet_v1_1.0_128", 
            "mobilenet_v1_0.75_224", "mobilenet_v1_0.75_192", "mobilenet_v1_0.75_160", "mobilenet_v1_0.75_128",
            "mobilenet_v1_0.5_224", "mobilenet_v1_0.5_192", "mobilenet_v1_0.5_160", "mobilenet_v1_0.5_128",
            "mobilenet_v1_0.25_224", "mobilenet_v1_0.25_192", "mobilenet_v1_0.25_160", "mobilenet_v1_0.25_128",
            ]

dpuList = ["B4096", "B3136", "B2304", "B1600", "B1152", "B1024", "B800", "B512"]

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="mobilenet_v1_1.0_224",choices=modelList, help="Default: MODEL=mobilenet_v1_1.0_224")	
parser.add_argument("-d", "--dpu", type=str, default="B4096",choices=dpuList , help="Default: DPU=B4096") # default="arch_myDPU.json"
parser.add_argument("-q", "--quantize", action='store_true', help="If you want start the quantization use python VitisAI_quantize_and_compile_no_zoo_model_flow.py -q")
parser.add_argument("-c", "--compile", action='store_true')
parser.add_argument("-o", "--outputLayer", type=str, default="MobilenetV1/Predictions/Reshape") # MobilenetV1/Predictions/Reshape_1
args = parser.parse_args()

print("************************************")
print("INPUT PARAMETERS:")
print(f"\tmodel: {args.model}")
print(f"\tDPU: {args.dpu}")
print(f"\tOutput Layer: {args.outputLayer}")
print(f"\tExecute quantization: {args.quantize}")
print(f"\tExecute compilation: {args.compile}")
print("************************************")

random.seed(42)

path = "./tf_models"
input_models = f"{args.model}/{args.model}_frozen.pb"
input_frozen_graph = os.path.join(path, input_models)

t0 = time.time()
stream = os.popen(f'vai_q_tensorflow inspect --input_frozen_graph={input_frozen_graph}')
output = stream.read()
print(output)
t1 = time.time()

if args.quantize:
    imageSize = int(args.model.split("_")[3])
    t6 = time.time()
    load_and_preprocess(imageSize)
    t7 = time.time()
    print(f"Loading, preprocessing and storing images time: {round(t7-t6, 3)} s")

    shell_command = f"vai_q_tensorflow quantize \
                        --input_frozen_graph ./tf_models/{args.model}/{args.model}_frozen.pb \
                        --input_fn input_func.calib_input \
                        --output_dir ./vai_quant_models/{args.model} \
                        --input_nodes input \
                        --output_nodes {args.outputLayer} \
                        --input_shapes ?,{imageSize},{imageSize},3 \
                        --calib_iter 32"

    t2 = time.time()
    stream = os.popen(shell_command)
    output = stream.read()
    print(output)
    t3 = time.time()

if args.compile:
    if os.path.exists(f"./vai_compiled_models/{args.dpu}") is False:
        os.mkdir(f"./vai_compiled_models/{args.dpu}")

    shell_command = f"vai_c_tensorflow \
                        --frozen_pb ./vai_quant_models/{args.model}/quantize_eval_model.pb \
                        --arch ./Arch_files/arch_{args.dpu}.json \
                        --output_dir ./vai_compiled_models/{args.dpu}/{args.model}_{args.dpu} \
                        --net_name {args.model}_{args.dpu}"

    t4 = time.time()
    stream = os.popen(shell_command)
    output = stream.read()
    print(output)
    t5 = time.time()


print("Script timing:")
print(f"\tInspecting frozen graph time: {round(t1-t0, 3)} s")
if args.quantize:
    print(f"\tQuantize time: {round(t3-t2, 3)} s")
if args.compile:
    print(f"\tCompiling time: {round(t5-t4, 3)} s")






# image = cv2.imread("dog_224x224.png")
# # print("image.dtype: ", image.dtype)
# # print("image.dtype.type: ", image.dtype.type)

# image = preprocess.resize_shortest_edge(image, imageSize)
# image = preprocess.central_crop(image, imageSize, imageSize)
# image = preprocess.tf_imagenet_preprocess(image)

# # print("image.dtype: ", image.dtype)
# # print("image.dtype.type: ", image.dtype.type)
# # sys.exit(1)

# print(image.shape)
# image = np.expand_dims(image, axis=0)
# print(image.shape)

# np.savez('./calib_data.npz', data = image)
