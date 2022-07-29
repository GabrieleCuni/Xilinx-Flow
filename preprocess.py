from configparser import Interpolation
import cv2
import numpy as np
import tensorflow as tf

def resize_shortest_edge(image, size):
    H, W = image.shape[:2]
    if H >= W:
        nW = size
        nH = int(float(H)/W * size)
    else:
        nH = size
        nW = int(float(W)/H * size)
    return cv2.resize(image,(nW,nH), interpolation=cv2.INTER_LINEAR)

def central_crop(image, crop_height, crop_width):
    image_height = image.shape[0]
    image_width = image.shape[1]
    offset_height = (image_height - crop_height) // 2
    offset_width = (image_width - crop_width) // 2
    return image[offset_height:offset_height + crop_height, offset_width:offset_width + crop_width, :]

def central_crop_percent(image, fraction=0.875):
    image_height = image.shape[0]
    image_width = image.shape[1]
    crop_height = int(image_height * fraction)
    crop_width = int(image_width * fraction)
    offset_height = (image_height - crop_height) // 2
    offset_width = (image_width - crop_width) // 2
    return image[offset_height:offset_height + crop_height, offset_width:offset_width + crop_width, :]

def tf_imagenet_preprocess(x):
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(float, copy=False)

    x /= 127.5
    x -= 1.
    return x

def finalPreprocess(image, height, width):
    image = central_crop_percent(image)
    image = cv2.resize(image,(width,height), interpolation=cv2.INTER_LINEAR)
    image = tf_imagenet_preprocess(image)

    return image

def desperatePreprocess(image, height, width): 
    image = central_crop_percent(image)
    image = cv2.resize(image,(width,height), interpolation=cv2.INTER_LINEAR)
    image = np.subtract(image, 0.5)
    image = np.multiply(image, 2.0)

    return image


# image, 224,224. The other params must be the default ones.
def preprocess_for_eval(image,
                        height,
                        width,
                        central_fraction=0.875,
                        scope=None,
                        central_crop=True,
                        use_grayscale=False):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        # image = tf.convert_to_tensor(image)
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if use_grayscale:
            image = tf.image.rgb_to_grayscale(image)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_crop and central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
            image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)

        a = image.eval(session=tf.Session())
        a = np.expand_dims(a,0)
        return a
        # return image



