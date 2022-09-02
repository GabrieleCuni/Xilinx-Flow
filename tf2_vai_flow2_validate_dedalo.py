import os 
import argparse
import time

import tensorflow as tf

"""
./docker_run.sh xilinx/vitis-ai:1.3.411

Attention to 1/31 [..............................] - ETA: 0sKilled that means the RAM is not enougth
"""

def getTFdataset(imageSize):
    path = os.path.join("/","space_fast","imagenet_folder","val")

    batchedDataset = tf.keras.utils.image_dataset_from_directory(
                path,
                labels='inferred',
                label_mode='int',
                class_names=None,
                color_mode='rgb',
                batch_size=None,
                image_size=(imageSize, imageSize),
                shuffle=True,
                seed=42,
                validation_split=None,
                subset=None,
                interpolation='bilinear',
                follow_links=False,
                crop_to_aspect_ratio=False
            )

    return batchedDataset


def validation(alpha, imageSize):
    batchedValidationDataset = getTFdataset(imageSize)

    print("Start Validation of the quantized model")
    t0 = time.time()
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
    batchedDataset = getTFdataset(imageSize)

    model.compile(	
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics= [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)]) 
    print("Start validate original tf2 model")
    print("Validation accuracy:")
    t0 = time.time()
    model.evaluate(batchedDataset, verbose=2)
    t1 = time.time()
    print(f"Stop Validation. Time: {t1-t0}")    

def main():
    alphaChoices = [1.0, 0.75, 0.5, 0.25]
    imageSizeChoices = [224, 192, 160, 128]
    dpuChoices = ["B4096", "B3136", "B2304", "B1600", "B1152", "B1024", "B800", "B512"]
    tfModelsPath = os.path.join("tf_models")

    if not os.path.exists(tfModelsPath):
        os.mkdir(tfModelsPath)    

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--alpha", type=float, default=1.0, choices=alphaChoices, help="Default: 1.0")	
    parser.add_argument("-s", "--imageSize", type=int, default=224, choices=imageSizeChoices, help="Default: 224")
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("--validate", action="store_true", help="If you want to validate the quantized model --validate") 
    parser.add_argument("--validateOriginal", action="store_true", help="If you want to validate the original tf2 model --validateOriginal")
    args = parser.parse_args()

    print("************************************")
    print("INPUT PARAMETERS:")
    print(f"\tmodel: mobilenet_v1_{args.alpha}_{args.imageSize}")
    print(f"\tVerbose: {args.verbose}")
    print(f"\tExecute original model validation: {args.validateOriginal}")
    print(f"\tExecute quantized model validation: {args.validate}")
    print("************************************")

    # model = tf.keras.applications.MobileNet(alpha=args.alpha, input_shape=(args.imageSize,args.imageSize,3))

    if args.verbose:
        if model is None:
            model = tf.keras.applications.MobileNet(alpha=args.alpha, input_shape=(args.imageSize,args.imageSize,3))
        print(model.summary())

    if args.validateOriginal:
        if model is None:
            model = tf.keras.applications.MobileNet(alpha=args.alpha, input_shape=(args.imageSize,args.imageSize,3))
        validateOriginal(model, args.imageSize)


    if args.validate:
        validation(args.alpha, args.imageSize)    

    

if __name__ == "__main__":
    main()