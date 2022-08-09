import os 
import argparse
import time
import sys
from myUtils import DatasetGenerator
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize

"""
./docker_run.sh xilinx/vitis-ai:1.3.411

Attention to 1/31 [..............................] - ETA: 0sKilled that means the RAM is not enougth
"""

def getDataset(path, imageSize, start, stop):
    datasetPath = os.path.join(path, f"dataset_{imageSize}_{start}_{stop}")
    if not os.path.exists(datasetPath):
        t0 = time.time()
        print("\nStart make dataset")
        print(path)
        print(f"Start image index: {start}")
        print(f"Stop image index: {stop}")
        print(f"imageSize: {imageSize}")
        datasetGenerator = DatasetGenerator(batch_size=32, startImageNumber=start, stopImageNumber=stop, width=imageSize, height=imageSize)
        batchedDataset = datasetGenerator.make_dataset()
        # print(f"Number of images: {len(batchedDataset)}")
        # print(f"Dataset spec: {batchedDataset.element_spec}") # (TensorSpec(shape=(32,), dtype=tf.float32, name=None), TensorSpec(shape=(32,), dtype=tf.float32, name=None))
        print("\nStop make dataset")
        print("Saving dataset on disk")
        tf.data.experimental.save(batchedDataset, datasetPath)
        t1 = time.time()
        print(f"Dataset saved on disk. Time: {t1-t0}")
        
    else:
        print("Loading Dataset from disk")
        t0 = time.time()
        batchedDataset = tf.data.experimental.load(datasetPath, element_spec=(tf.TensorSpec(shape=[32,imageSize,imageSize,3], dtype=tf.float32), tf.TensorSpec(shape=[32,1], dtype=tf.float32)))
        print(f"Number of images: {len(batchedDataset)}")
        t1 = time.time()
        print(f"Dataset loaded. Time: {t1-t0}")

    return batchedDataset

def quantization(model, preprocessQuantDataPath, alpha, imageSize, start, stop):
    batchedQuantDataset = getDataset(preprocessQuantDataPath, imageSize, start, stop)

    print("Start Quantization")
    t0 = time.time()
    quantizer = vitis_quantize.VitisQuantizer(model)
    quantized_model = quantizer.quantize_model(calib_dataset=batchedQuantDataset)
    quantized_model.save(os.path.join("tf2_vai_quant_models",f"quantized_mobilenet_{alpha}_{imageSize}.h5"))
    t1 = time.time()
    print(f"Stop Quantization. Time: {t1-t0}")

def validation(preprocessValDataPath, alpha, imageSize, start, stop):
    batchedValidationDataset = getDataset(preprocessValDataPath, imageSize, start, stop)

    print("Start Validation")
    t0 = time.time()
    with vitis_quantize.quantize_scope():
        modelPath = os.path.join("tf2_vai_quant_models",f"quantized_mobilenet_{alpha}_{imageSize}.h5")
        quantized_model = tf.keras.models.load_model(modelPath)

    quantized_model.compile(	
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics= tf.keras.metrics.SparseTopKCategoricalAccuracy()) # Attenzione sto facendo il top 5 accuracy!!!
    print("Validation accuracy:")
    quantized_model.evaluate(batchedValidationDataset, verbose=2)
    t1 = time.time()
    print(f"Stop Validation. Time: {t1-t0}")
    

def main():
    alphaChoices = [1.0, 0.75, 0.5, 0.25]
    imageSizeChoices = [224, 192, 160, 128]
    dpuChoices = ["B4096", "B3136", "B2304", "B1600", "B1152", "B1024", "B800", "B512"]
    tfModelsPath = os.path.join("tf_models")
    preprocessValDataPath = os.path.join("tf2_preprocessDatasets", "validationDatasets")
    preprocessQuantDataPath = os.path.join("tf2_preprocessDatasets", "quantizationDatasets")

    if not os.path.exists(tfModelsPath):
        os.mkdir(tfModelsPath)

    if not os.path.exists(preprocessValDataPath):
        os.makedirs(preprocessValDataPath)

    if not os.path.exists(preprocessQuantDataPath):
        os.makedirs(preprocessQuantDataPath)
    

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alpha", type=float, default=1.0, choices=alphaChoices, help="Default: 1.0")	
    parser.add_argument("-s", "--imageSize", type=int, default=224, choices=imageSizeChoices, help="Default: 224")
    parser.add_argument("--startQuant", type=int, default=0)
    parser.add_argument("--stopQuant", type=int, default=1024)
    parser.add_argument("--startVal", type=int, default=1024)
    parser.add_argument("--stopVal", type=int, default=2048)
    parser.add_argument("-d", "--dpu", type=str, default="B4096",choices=dpuChoices , help="Default: B4096")
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("-q", "--quantize", action='store_true', help="If you want start the quantization -q")
    parser.add_argument("--validate", action="store_true", help="If you want to validate --validate")
    parser.add_argument("-c", "--compile", action='store_true', help="If you want start the compilation -q")
    # parser.add_argument("-o", "--outputLayer", type=str, default="MobilenetV1/Predictions/Reshape") # MobilenetV1/Predictions/Reshape_1
    args = parser.parse_args()

    print("************************************")
    print("INPUT PARAMETERS:")
    print(f"\tmodel: mobilenet_v1_{args.alpha}_{args.imageSize}")
    print(f"\tDPU: {args.dpu}")
    # print(f"\tOutput Layer: {args.outputLayer}")
    print(f"\tExecute quantization: {args.quantize}")
    print(f"\tExecute compilation: {args.compile}")
    print("************************************")

    model = tf.keras.applications.MobileNet(alpha=args.alpha, input_shape=(args.imageSize,args.imageSize,3))

    modelPath = os.path.join(tfModelsPath, f"tf2_mobilenet_v1_{args.alpha}_{args.imageSize}")
    if not os.path.exists(modelPath):
        model.save(modelPath)
        model = tf.keras.applications.MobileNet(alpha=args.alpha, input_shape=(args.imageSize,args.imageSize,3))
    else:
        model = tf.keras.models.load_model(modelPath, compile=False)

    if args.verbose:
        print(model.summary())

    if args.quantize:        
        quantization(model, preprocessQuantDataPath, args.alpha, args.imageSize, args.startQuant, args.stopQuant)

    if args.validate:
        validation(preprocessValDataPath, args.alpha, args.imageSize, args.startVal, args.stopVal)
    

if __name__ == "__main__":
    main()