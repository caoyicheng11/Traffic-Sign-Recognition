import cv2
import math
import random
import numpy as np

class BaseTransform():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485*255, 0.456*255, 0.406*255]
        if std is None:
            std = [0.229*255, 0.224*255, 0.225*255]
        self.mean = np.array(mean).reshape((1, 3, 1, 1))
        self.std = np.array(std).reshape((1, 3, 1, 1))

    def __call__(self, x):
        return (x - self.mean) / self.std

class RandomRotate(object):
    def __init__(self, prob=0.5, degree=10):
        self.prob = prob
        self.degree = degree

    def __call__(self, imgs):
        if random.uniform(0, 1) >= self.prob:
            return imgs
        else:
            dh, dw = imgs.shape[-2:]
            # rotation
            degree = random.uniform(-self.degree, self.degree)
            M1 = cv2.getRotationMatrix2D((dh // 2, dw // 2), degree, 1)
            # affine
            if len(imgs.shape) == 4:
                imgs = imgs.transpose(0, 2, 3, 1)
            imgs = [cv2.warpAffine(_[0, ...], M1, (dw, dh))
                for _ in np.split(imgs, imgs.shape[0], axis=0)]
            imgs = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in imgs], 0)
            if len(imgs.shape) == 4:
                imgs = imgs.transpose(0, 3, 1, 2)
            return imgs

class RandomBrightness(object):
    def __init__(self, prob=0.5, delta=(-0.1, 0.1)):
        """
        Args:
            prob (float): Probability of applying brightness adjustment.
            delta (tuple): Range of brightness change (lower, upper).
                           A positive value increases brightness, and a negative value decreases brightness.
        """
        self.prob = prob
        self.delta = delta  # delta is a tuple (lower, upper) for brightness change

    def __call__(self, imgs):
        """
        Args:
            imgs (numpy.ndarray): Input images with shape (n, c, h, w).
        Returns:
            numpy.ndarray: Brightness-adjusted images.
        """
        if random.uniform(0, 1) >= self.prob:
            return imgs
        else:
            # Randomly sample a brightness change within the delta range
            delta = random.uniform(self.delta[0], self.delta[1])
            # Apply the same delta to all channels and pixels
            imgs = imgs + delta * 255
            # Clip the values to ensure they are within the valid range [0, 255]
            imgs = np.clip(imgs, 0, 255)
            return imgs

class RandomBlur(object):
    def __init__(self, prob=0.5, size=5):
        """
        Args:
            prob (float): Probability of applying blur.
            size (int): Maximum size of the Gaussian kernel (must be odd).
        """
        self.prob = prob
        self.size = size

    def __call__(self, imgs):
        """
        Args:
            imgs (numpy.ndarray): Input images of shape (N, C, H, W) or (C, H, W).
        Returns:
            numpy.ndarray: Transformed images.
        """
        if random.uniform(0, 1) >= self.prob:
            return imgs
        else:
            # Generate a random odd kernel size between 1 and size
            kernel_size = random.randrange(1, self.size + 1, 2)
            # Apply Gaussian blur
            if len(imgs.shape) == 4:  # Batch of images (N, C, H, W)
                imgs = imgs.transpose(0, 2, 3, 1)  # Change to (N, H, W, C) for OpenCV
                imgs = np.array([cv2.GaussianBlur(img, (kernel_size, kernel_size), 0) 
                                 for img in imgs])
                imgs = imgs.transpose(0, 3, 1, 2)  # Change back to (N, C, H, W)
            elif len(imgs.shape) == 3:  # Single image (C, H, W)
                imgs = imgs.transpose(1, 2, 0)  # Change to (H, W, C) for OpenCV
                imgs = cv2.GaussianBlur(imgs, (kernel_size, kernel_size), 0)
                imgs = imgs.transpose(2, 0, 1)  # Change back to (C, H, W)
            return imgs