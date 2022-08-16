import argparse
from pyexpat import features
import time
import os
import tensorflow as tf


def getDatasetWithKeras(imageSize, quantSize, validationSize):
    t0 = time.time()
    print("\nStart make dataset with Keras")

    path = os.path.join("calibration_images", "val")    

    dataset = tf.keras.utils.image_dataset_from_directory(
                    path,
                    labels='inferred',
                    label_mode='int',
                    class_names=None,
                    color_mode='rgb',
                    batch_size=None,
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
    t1 = time.time()
    print(f"Stop make dataset with Keras. Time: {t1-t0}")

    return quantDataset, validationDataset


def makeTfRecord(output_filename, dataset):
    print("Start write TFrecord")
    count = 0
    with tf.io.TFRecordWriter(output_filename) as writer:
        for x,y in dataset:
            count += 1
            # if count == 0:
            #     print("raw image: ", x)
            #     print("label: ", y)
            #     count = 1
            label_features = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(y)]))
            image_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x).numpy()]))
            mapping = {
                "label": label_features,
                "image_raw": image_feature
            }
            tf_example = tf.train.Example(features=tf.train.Features(feature=mapping))
            writer.write(tf_example.SerializeToString())
    
    print(f"Stop write TFrecord of {count} elements")

 

def main():
    imageSizeChoices = [224, 192, 160, 128]

    parser = argparse.ArgumentParser()	
    parser.add_argument("-s", "--imageSize", type=int, default=224, choices=imageSizeChoices, help="Default: 224")
    parser.add_argument("-o", "--output", type=str, default="dataset.tfrecord")
    parser.add_argument("--quantSize", type=int, default=1024)
    parser.add_argument("--validationSize", type=int, default=1024)
    args = parser.parse_args()

    quantDataset, validationDataset = getDatasetWithKeras(args.imageSize, args.quantSize, args.validationSize)

    makeTfRecord(args.output, quantDataset)

    # quantDaasetPath = os.path.join("tf2_datasets","quantization")
    # validationDatasetPath =  os.path.join("tf2_datasets","validation")

    # if not os.path.exists(quantDaasetPath):
    #     os.makedirs(quantDaasetPath)

    # if not os.path.exists(validationDatasetPath):
    #     os.makedirs(validationDatasetPath)


if __name__ == "__main__":
    main()