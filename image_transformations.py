import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch

class ImageTransformations:
    def __init__(self):
        pass

    @staticmethod
    def rotate(image, angle=None):
        if angle is None:
            angle = random.uniform(-30, 30)
        return image.rotate(angle)

    @staticmethod
    def scale(image, scale_factor=None):
        if scale_factor is None:
            scale_factor = random.uniform(0.8, 1.2)
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return image.resize((new_width, new_height), Image.BICUBIC)

    @staticmethod
    def flip(image, horizontal=True):
        if horizontal:
            return ImageOps.mirror(image)
        else:
            return ImageOps.flip(image)

    @staticmethod
    def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
        transform = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        return transform(image)

    @staticmethod
    def elastic_transform(image, alpha=34, sigma=4):
        random_state = np.random.RandomState(None)
        image = np.array(image)

        shape = image.shape
        dx = random_state.rand(*shape) * 2 - 1
        dy = random_state.rand(*shape) * 2 - 1
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

        indices = np.reshape(y + dy * alpha, (-1, 1)), np.reshape(x + dx * alpha, (-1, 1)), np.reshape(z, (-1, 1))
        distorted_image = np.zeros_like(image)
        map_coordinates(image, indices, output=distorted_image, order=1, mode='reflect')
        return Image.fromarray(distorted_image)

def map_coordinates(input_array, coordinates, order=1, mode='reflect'):
    from scipy.ndimage import map_coordinates
    return map_coordinates(input_array, coordinates, order=order, mode=mode)


def do_transformations(img, transformation_type='all'):
    transformer = ImageTransformations()

    if transformation_type == 'rotate':
        return transformer.rotate(img)

    elif transformation_type == 'scale':
        return transformer.scale(img)

    elif transformation_type == 'flip':
        return transformer.flip(img)

    elif transformation_type == 'color_jitter':
        return transformer.color_jitter(img)

    elif transformation_type == 'elastic_transform':
        return transformer.elastic_transform(img)

    elif transformation_type == 'all':
        img = transformer.rotate(img)
        img = transformer.scale(img)
        img = transformer.flip(img)
        img = transformer.color_jitter(img)
        return transformer.elastic_transform(img)