'''
This file contains all the functions used to produce the colourspaces
and coloured edge maps mentioned in our paper "Coloured Edge Maps for
Oil Palm Ripeness Classification"

All functions expect input images in the BGR colourspace.
'''

import cv2
import numpy as np


# Colourspaces
def to_YCrCb(image):
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2YCrCb)
    image = preprocessing_for_arch(image)
    return image

# Coloured edge maps
def gradient_intensity(image, filter_x, filter_y):
    grad_x = cv2.filter2D(image, -1, filter_x)
    grad_y = cv2.filter2D(image, -1, filter_y)
    final_image = np.hypot(grad_x, grad_y)
    normalized = final_image / final_image.max() * 255
    return normalized.astype(np.uint8)

def to_sobel_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    kernel_x = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    kernel_y = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    image = gradient_intensity(image, kernel_x, kernel_y)
    image = preprocessing_for_arch(image)
    return image

def to_LoG_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    kernel = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
    image = cv2.GaussianBlur(image, (5,5), 0)
    image = cv2.filter2D(image.astype(np.float32), -1, kernel)
    image = image / image.max() * 255
    image = preprocessing_for_arch(image)
    return image.astype(np.uint8)

def to_prewitt_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    kernel_x = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernel_y = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    image = gradient_intensity(image, kernel_x, kernel_y)
    image = preprocessing_for_arch(image)
    return image

def to_kirsch_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    kernel_1 = np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])
    kernel_2 = np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])
    kernel_3 = np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
    kernel_4 = np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])
    kernel_5 = np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])
    kernel_6 = np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])
    kernel_7 = np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])
    kernel_8 = np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])
    grad_1 = cv2.normalize(cv2.filter2D(image, cv2.CV_32F, kernel_1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    grad_2 = cv2.normalize(cv2.filter2D(image, cv2.CV_32F, kernel_2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    grad_3 = cv2.normalize(cv2.filter2D(image, cv2.CV_32F, kernel_3), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    grad_4 = cv2.normalize(cv2.filter2D(image, cv2.CV_32F, kernel_4), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    grad_5 = cv2.normalize(cv2.filter2D(image, cv2.CV_32F, kernel_5), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    grad_6 = cv2.normalize(cv2.filter2D(image, cv2.CV_32F, kernel_6), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    grad_7 = cv2.normalize(cv2.filter2D(image, cv2.CV_32F, kernel_7), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    grad_8 = cv2.normalize(cv2.filter2D(image, cv2.CV_32F, kernel_8), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    image = np.maximum.reduce([grad_1, grad_2, grad_3, grad_4, grad_5, grad_6, grad_7, grad_8])
    image = image / image.max() * 255
    image = preprocessing_for_arch(image)
    return image.astype(np.uint8)