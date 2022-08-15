import argparse
import time
import os
import tensorflow as tf


def getDatasetWithKeras(imageSize, quantSize, validationSize):
    t0 = time.time()
    print("\nStart make dataset with Keras")

    path = os.path.join("calibration_images", "val")

    quantDaasetPath = os.path.join("tf2_datasets","quantization")
    validationDatasetPath =  os.path.join("tf2_datasets","validation")

    if not os.path.exists(quantDaasetPath):
        os.makedirs(quantDaasetPath)

    if not os.path.exists(validationDatasetPath):
        os.makedirs(validationDatasetPath)
    

    dataset = tf.keras.utils.image_dataset_from_directory(
                    path,
                    labels='inferred',
                    label_mode='int',
                    class_names=None,
                    color_mode='rgb',
                    batch_size=32,
                    image_size=(imageSize, imageSize),
                    shuffle=True,
                    seed=None,
                    validation_split=None,
                    subset=None,
                    interpolation='bilinear',
                    follow_links=False,
                    crop_to_aspect_ratio=False
                )


    quantDataset = dataset.take(quantSize) 
    dataset = dataset.skip(quantSize)
    validationDataset = dataset.take(validationSize)
    tf.data.experimental.save(quantDataset, quantDaasetPath)
    tf.data.experimental.save(validationDataset, validationDatasetPath)


    t1 = time.time()
    print(f"Stop make dataset with Keras. Time: {t1-t0}")
 

def main():
    imageSizeChoices = [224, 192, 160, 128]



    parser = argparse.ArgumentParser()	
    parser.add_argument("-s", "--imageSize", type=int, default=224, choices=imageSizeChoices, help="Default: 224")
    parser.add_argument("--quantSize", type=int, default=1024)
    parser.add_argument("--validationSize", type=int, default=1024)
    args = parser.parse_args()

    getDatasetWithKeras(args.imageSize, args.quantSize, args.validationSize)


if __name__ == "__main__":
    main()