import cv2
import numpy as np

def resize_shortest_edge(image, size):
    H, W = image.shape[:2]
    if H >= W:
        nW = size
        nH = int(float(H)/W * size)
    else:
        nH = size
        nW = int(float(W)/H * size)
    return cv2.resize(image,(nW,nH))

def central_crop(image, crop_height, crop_width):
    image_height = image.shape[0]
    image_width = image.shape[1]
    offset_height = (image_height - crop_height) // 2
    offset_width = (image_width - crop_width) // 2
    return image[offset_height:offset_height + crop_height, offset_width:
                offset_width + crop_width, :]

def tf_imagenet_preprocess(x):
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(float, copy=False)

    x /= 127.5
    x -= 1.
    return x

