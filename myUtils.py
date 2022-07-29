import cv2
import os 
import numpy as np
import tensorflow as tf


class DatasetGenerator:
    def __init__(self, batch_size, imageNumber, width, height):
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.imageNumber = imageNumber

    def _getImageNames(self):
        imageNames = os.listdir(os.path.join("calibration_images","ILSVRC2012_img_val"))
        imageNames = imageNames[:self.imageNumber]

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
        if not issubclass(x.dtype.type, np.floating):
            x = x.astype(float, copy=False)
        x /= 127.5
        x -= 1.
        return x

    def make_dataset(self):
        imageNames = self._getImageNames()
        preprocessed_images = []
        for imageName in imageNames:            
            image = cv2.imread(os.path.join("calibration_images","ILSVRC2012_img_val",imageName))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self._central_crop(image)
            image = self._standardization(image)
            if image.shape == (224, 224, 3):
                preprocessed_images.append(image)
                
        dataset = tf.data.Dataset.from_tensor_slices(preprocessed_images)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        return dataset