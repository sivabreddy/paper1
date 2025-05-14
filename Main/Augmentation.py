# Image augmentation functions including rotation and cropping operations
# Uses OpenCV and NumPy for image transformations

import cv2, numpy as np
import matplotlib.pyplot as plt

def Augmentation(input_im):
    """
    Performs image augmentation including rotation and cropping
    Args:
        input_im: Input image to be augmented (numpy array)
    Returns:
        tuple: (rotated_image, cropped_image) - both resized to 256x256
    """

    ################ rotating #################
    # Get image dimensions and set rotation angles (30 and 10 degrees converted to radians)
    rows, cols, dim = input_im.shape
    angle,angle = np.radians(30),np.radians(10)
    
    # Create 3x3 transformation matrix for rotation
    # [cosθ  -sinθ  0]
    # [sinθ   cosθ  0]
    # [0      0     1]
    M = np.float32([[np.cos(angle), -(np.sin(angle)), 0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
    rotated_img = cv2.warpPerspective(input_im, M, (int(cols), int(rows)))
    rotated_img = cv2.resize(rotated_img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    height, width = input_im.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 50, .5)

    ############### cropping ##################
    # Crop 30% from top and left of image (remove 30% of width/height from start)
    cropped_image = input_im[int(cols*0.3): , int(cols*0.3):] #30%
    cropped_image = cv2.resize(cropped_image, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    return  rotated_img, cropped_image