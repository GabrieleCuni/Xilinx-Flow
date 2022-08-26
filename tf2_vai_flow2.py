import os 
import argparse
from random import seed
import time
import sys

import numpy as np
from myUtils import DatasetGenerator
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize

"""
./docker_run.sh xilinx/vitis-ai:1.3.411

Attention to 1/31 [..............................] - ETA: 0sKilled that means the RAM is not enougth
"""

def _normalization(x, y):
    x /= 127.5
    x -= 1.
    return x, y


def _parser_fn(record):
    name_to_features = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'imageSize': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    return tf.io.parse_single_example(record, name_to_features)


def _decode_fn(record):
    # image = tf.io.decode_raw(record['image_raw'], out_type=tf.float32)
    image = tf.io.parse_tensor(record['image_raw'], out_type=tf.float32)
    label = record['label']
    dimension = record['imageSize']
    image = tf.reshape(image, (dimension, dimension, 3))
    return (image, label)


def getTFdataset(imageSize, subset):
    dataset = tf.data.TFRecordDataset(os.path.join("tf2_datasets", f"{subset}_{imageSize}.tfrecord"))
    dataset = dataset.map(_parser_fn)
    dataset = dataset.map(_decode_fn)
    dataset = dataset.map(_normalization)
    batchedDataset = dataset.batch(32, drop_remainder=True)

    return batchedDataset


def quantization(model, alpha, imageSize):
    batchedQuantDataset = getTFdataset(imageSize, "quantization")

    print("Start Quantization")
    t0 = time.time()
    quantizer = vitis_quantize.VitisQuantizer(model)
    quantized_model = quantizer.quantize_model(calib_dataset=batchedQuantDataset)
    quantized_model.save(os.path.join("tf2_vai_quant_models",f"quantized_mobilenet_{alpha}_{imageSize}.h5"))
    t1 = time.time()
    print(f"Stop Quantization. Time: {t1-t0}")


def validation(alpha, imageSize):
    batchedValidationDataset = getTFdataset(imageSize, "validation")

    print("Start Validation of the quantized model")
    t0 = time.time()
    with vitis_quantize.quantize_scope():
        modelPath = os.path.join("tf2_vai_quant_models",f"quantized_mobilenet_{alpha}_{imageSize}.h5")
        quantized_model = tf.keras.models.load_model(modelPath, compile=False)

    quantized_model.compile(	
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics= [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]) 
    print("Validation accuracy:")
    quantized_model.evaluate(batchedValidationDataset, verbose=2)
    t1 = time.time()
    print(f"Stop Validation. Time: {t1-t0}")

    
def validateOriginal(model, imageSize):
    # i = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
    # x = tf.cast(i, tf.float32)
    # x = tf.keras.applications.mobilenet.preprocess_input(x)
    # core = tf.keras.applications.MobileNet(alpha=alpha, input_shape=(imageSize,imageSize,3))
    # x = core(x)
    # model = tf.keras.Model(inputs=[i], outputs=[x])

    # datasetGenerator = DatasetGenerator(batch_size=32, startImageNumber=start, stopImageNumber=stop, width=imageSize, height=imageSize)
    # batchedDataset = datasetGenerator.make_dataset_without_preprocessing()

    batchedDataset = getTFdataset(imageSize, "validation")

    model.compile(	
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics= [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]) 
    print("Start validate original tf2 model")
    print("Validation accuracy:")
    t0 = time.time()
    model.evaluate(batchedDataset, verbose=2)
    t1 = time.time()
    print(f"Stop Validation. Time: {t1-t0}")

def compiler(dpu, alpha, imageSize):
    outputPath = os.path.join("tf2_vai_compiled_models", dpu, f"tf2_mobilenet_v1_{alpha}_{imageSize}_{dpu}")
    quantModelPath = os.path.join("tf2_vai_quant_models",f"quantized_mobilenet_{alpha}_{imageSize}.h5")
    archPath = os.path.join("Arch_files", f"arch_{dpu}.json")

    if os.path.exists(outputPath) is False:
        os.makedirs(outputPath)

    
    shell_command = f"vai_c_tensorflow2 \
                        -m {quantModelPath} \
                        -a {archPath} \
                        -o {outputPath} \
                        -n tf2mobilenet_v1_{alpha}_{imageSize}_{dpu}"   

    stream = os.popen(shell_command)
    output = stream.read()
    print(output)
    

def main():
    alphaChoices = [1.0, 0.75, 0.5, 0.25]
    imageSizeChoices = [224, 192, 160, 128]
    dpuChoices = ["B4096", "B3136", "B2304", "B1600", "B1152", "B1024", "B800", "B512"]
    tfModelsPath = os.path.join("tf_models")
    # preprocessValDataPath = os.path.join("tf2_preprocessDatasets", "validationDatasets")
    # preprocessQuantDataPath = os.path.join("tf2_preprocessDatasets", "quantizationDatasets")

    if not os.path.exists(tfModelsPath):
        os.mkdir(tfModelsPath)

    # if not os.path.exists(preprocessValDataPath):
    #     os.makedirs(preprocessValDataPath)

    # if not os.path.exists(preprocessQuantDataPath):
    #     os.makedirs(preprocessQuantDataPath)
    

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alpha", type=float, default=1.0, choices=alphaChoices, help="Default: 1.0")	
    parser.add_argument("-s", "--imageSize", type=int, default=224, choices=imageSizeChoices, help="Default: 224")
    parser.add_argument("-d", "--dpu", type=str, default="B4096",choices=dpuChoices , help="Default: B4096")
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("-q", "--quantize", action='store_true', help="If you want start the quantization -q")
    parser.add_argument("--validate", action="store_true", help="If you want to validate the quantized model --validate") 
    parser.add_argument("--validateOriginal", action="store_true", help="If you want to validate the original tf2 model --validateOriginal")
    parser.add_argument("-c", "--compile", action='store_true', help="If you want start the compilation -q")
    args = parser.parse_args()

    print("************************************")
    print("INPUT PARAMETERS:")
    print(f"\tmodel: mobilenet_v1_{args.alpha}_{args.imageSize}")
    print(f"\tDPU: {args.dpu}")
    print(f"\tVerbose: {args.verbose}")
    print(f"\tExecute original model validation: {args.validateOriginal}")
    print(f"\tExecute quantization: {args.quantize}")
    print(f"\tExecute quantized model validation: {args.validate}")
    print(f"\tExecute compilation: {args.compile}")
    print("************************************")

    model = tf.keras.applications.MobileNet(alpha=args.alpha, input_shape=(args.imageSize,args.imageSize,3))

    # modelPath = os.path.join(tfModelsPath, f"tf2_mobilenet_v1_{args.alpha}_{args.imageSize}")
    # if not os.path.exists(modelPath):
    #     model.save(modelPath)
    #     model = tf.keras.applications.MobileNet(alpha=args.alpha, input_shape=(args.imageSize,args.imageSize,3))
    # else:
    #     model = tf.keras.models.load_model(modelPath, compile=False)

    if args.verbose:
        print(model.summary())

    if args.validateOriginal:
        validateOriginal(model, args.imageSize)

    if args.quantize:        
        quantization(model, args.alpha, args.imageSize)

    if args.validate:
        validation(args.alpha, args.imageSize)    

    if args.compile:
        compiler(args.dpu, args.alpha, args.imageSize)
    

if __name__ == "__main__":
    main()