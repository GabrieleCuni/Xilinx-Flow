import argparse
from pyexpat import features
import time
import os
import tensorflow as tf

def getTotalDataset(imageSize):
    t0 = time.time()
    print("\nStart make total dataset with Keras")

    path = os.path.join("calibration_images", "val")    

    batchDataset = tf.keras.utils.image_dataset_from_directory(
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

    t1 = time.time()
    print(f"Stop make total dataset with Keras. Time: {t1-t0}")

    return batchDataset


def getDatasetWithKeras(imageSize, quantSize, validationSize):
    t0 = time.time()
    print("\nStart make dataset with Keras")

    path = os.path.join("calibration_images", "val")    

    quantDataset = tf.keras.utils.image_dataset_from_directory(
                    path,
                    labels='inferred',
                    label_mode='int',
                    class_names=None,
                    color_mode='rgb',
                    batch_size=None,
                    image_size=(imageSize, imageSize),
                    shuffle=True,
                    seed=42,
                    validation_split=0.5,
                    subset="training",
                    interpolation='bilinear',
                    follow_links=False,
                    crop_to_aspect_ratio=False
                )

    validationDataset = tf.keras.utils.image_dataset_from_directory(
                path,
                labels='inferred',
                label_mode='int',
                class_names=None,
                color_mode='rgb',
                batch_size=None,
                image_size=(imageSize, imageSize),
                shuffle=True,
                seed=42,
                validation_split=0.5,
                subset="validation",
                interpolation='bilinear',
                follow_links=False,
                crop_to_aspect_ratio=False
            )


    # quantDataset = dataset.take(quantSize) 
    # dataset = dataset.skip(quantSize)
    # validationDataset = dataset.take(validationSize)
    t1 = time.time()
    print(f"Stop make dataset with Keras. Time: {t1-t0}")

    return quantDataset.take(quantSize), validationDataset.take(validationSize)


def makeTfRecord(output_filename, dataset, imageSize):
    print("Start write TFrecord")
    count = 0
    with tf.io.TFRecordWriter(output_filename) as writer:
        for x,y in dataset:
            # if count == 0:
            #     print(x.dtype)
            count += 1
            
            label_features = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(y)]))
            imageSize_features = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(imageSize)]))
            image_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x).numpy()]))
            mapping = {
                "label": label_features,
                "imageSize": imageSize_features,
                "image_raw": image_feature
            }
            tf_example = tf.train.Example(features=tf.train.Features(feature=mapping))
            writer.write(tf_example.SerializeToString())
    
    print(f"Stop write TFrecord of {count} elements")

 

def main():
    imageSizeChoices = [224, 192, 160, 128]

    parser = argparse.ArgumentParser()	
    parser.add_argument("-s", "--imageSize", type=int, default=224, choices=imageSizeChoices, help="Default: 224")
    parser.add_argument("--quantSize", type=int, default=4096, help="Default: 4096")
    parser.add_argument("--validationSize", type=int, default=4096, help="Default: 4096")
    parser.add_argument("-t", "--total", action='store_true', help="Store all the dataset")
    args = parser.parse_args()

    print("************************************")
    print("INPUT PARAMETERS:")
    print(f"\tImage Size: {args.imageSize}")
    print(f"\tValidation dataset size: {args.validationSize}")
    print(f"\tQuantization dataset size: {args.quantSize}")
    print("************************************")

    datasetPath = os.path.join("tf2_datasets")

    if not os.path.exists(datasetPath):
        os.makedirs(datasetPath)

    if args.total is False:
        quantDataset, validationDataset = getDatasetWithKeras(args.imageSize, args.quantSize, args.validationSize)       

        path = os.path.join(datasetPath, f"quantization_{args.imageSize}.tfrecord")
        makeTfRecord(path, quantDataset, args.imageSize)

        path = os.path.join(datasetPath, f"validation_{args.imageSize}.tfrecord")
        makeTfRecord(path, validationDataset, args.imageSize)
    else:
        # path = os.path.join("/", "media", "gabriele", "Gabriele Hard Disk", f"totalDataset_{args.imageSize}.tfrecord")
        path = os.path.join(datasetPath, f"validation_{args.imageSize}.tfrecord")

        dataset = getTotalDataset(args.imageSize)
        makeTfRecord(path, dataset, args.imageSize)


if __name__ == "__main__":
    main()