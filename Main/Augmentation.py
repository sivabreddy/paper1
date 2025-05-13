import cv2, numpy as np
import  matplotlib.pyplot as plt

def Augmentation(input_im):

    ################ rotating #################
    rows, cols, dim = input_im.shape
    angle,angle = np.radians(30),np.radians(10)
    # transformation matrix for Rotation
    M = np.float32([[np.cos(angle), -(np.sin(angle)), 0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
    rotated_img = cv2.warpPerspective(input_im, M, (int(cols), int(rows)))
    rotated_img = cv2.resize(rotated_img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    height, width = input_im.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 50, .5)

    ############### cropping ##################
    cropped_image = input_im[int(cols*0.3): , int(cols*0.3):] #30%
    cropped_image = cv2.resize(cropped_image, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    return  rotated_img, cropped_image