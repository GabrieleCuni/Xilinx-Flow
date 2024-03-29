import cv2
import os 
import json
import numpy as np
import tensorflow as tf


class DatasetGenerator:
    def __init__(self, batch_size, startImageNumber, stopImageNumber, width, height):
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.startImageNumber = startImageNumber
        self.stopImageNumber = stopImageNumber

    def _getImageNames(self):
        imageNames = os.listdir(os.path.join("calibration_images","ILSVRC2012_img_val"))
        imageNames = imageNames[self.startImageNumber:self.stopImageNumber]

        return imageNames

    def _central_crop(self, image):
        crop_height = self.height
        crop_width = self.width

        image_height = image.shape[0]
        image_width = image.shape[1]
        offset_height = (image_height - crop_height) // 2
        offset_width = (image_width - crop_width) // 2
        return image[offset_height:offset_height + crop_height, offset_width:offset_width + crop_width, :]

    def _standardization(self, x):
        # if not issubclass(x.dtype.type, np.floating):
        #     x = x.astype(float, copy=False)
        x = tf.cast(x, tf.float32)
        x /= 127.5
        x -= 1.
        return x

    def _getImageNumber(self, image_name):
        return int(image_name.split("_")[2].split(".")[0])

    def _getMapping(self):
        GroudTruthFile = os.path.join("calibration_images","ILSVRC2012_validation_ground_truth.txt")
        with open(GroudTruthFile, "r") as f:
            mapping = f.readlines()
        return mapping

    def _getMapClslocMapping(self):
        mapClsloc = os.path.join("calibration_images","map_clsloc.txt")
        with open(mapClsloc, "r") as fp:
            mapClslocMapping = fp.readlines()
        return mapClslocMapping

    def _getImagenetClassIndex(self):
        with open(os.path.join("calibration_images","imagenet_class_index.json")) as fp:
            classIndex = json.load(fp)
        return classIndex

    def _getLabel(self, imageName):
        mapping = self._getMapping()
        mapClslocMapping = self._getMapClslocMapping()
        classIndexDict = self._getImagenetClassIndex()

        imageNumber = self._getImageNumber(imageName)
        groundTruth  = mapping[imageNumber-1]        
        stringToSplit = mapClslocMapping[int(groundTruth)-1]
        imagenetID = stringToSplit.split(" ")[0]
        label = list(classIndexDict.keys())[[i[0] for i in list(classIndexDict.values())].index((imagenetID))]

        # print(f"Name: {imageName}, Number: {imageNumber}, GroudTruth: {groundTruth}, imagenetID: {imagenetID}, label: {label}")

        return int(label)


    def make_dataset(self):
        imageNames = self._getImageNames()
        preprocessed_images = []
        labels = []
        for imageName in imageNames:       
            imagePath = os.path.join("calibration_images","ILSVRC2012_img_val",imageName)     
            # image = cv2.imread(imagePath)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = tf.io.read_file(imagePath)
            image = tf.io.decode_jpeg(image, channels=3)
            image = self._central_crop(image)
            image = self._standardization(image)
            if image.shape == (self.width, self.height, 3):
                preprocessed_images.append(image)
                labels.append(self._getLabel(imageName))
        print(f"MyUtils.py - Number of images: {len(preprocessed_images)}")
        dataset = tf.data.Dataset.from_tensor_slices((preprocessed_images, labels))
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        return dataset

    def make_dataset_without_preprocessing(self):
        imageNames = self._getImageNames()
        preprocessed_images = []
        labels = []
        for imageName in imageNames:       
            imagePath = os.path.join("calibration_images","ILSVRC2012_img_val",imageName)     
            image = tf.io.read_file(imagePath)
            image = tf.io.decode_jpeg(image, channels=3)
            # image = self._central_crop(image)    
            if image.shape[0] > self.height and image.shape[1] > self.height:   
                offset_height = (image.shape[0] - self.height) // 2
                offset_width = (image.shape[1] - self.width) // 2 
                image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, self.height, self.width)
    
                if image.shape == (self.width, self.height, 3):
                    preprocessed_images.append(image)
                    labels.append(self._getLabel(imageName))
        print(f"MyUtils.py - Number of images: {len(preprocessed_images)}")
        dataset = tf.data.Dataset.from_tensor_slices((preprocessed_images, labels))
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        return dataset