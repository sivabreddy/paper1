"""
Data Preparation Script
----------------------
Prepares medical image dataset by:
- Creating required directory structure
- Processing raw images and ground truth masks
- Converting ground truth to binary format
- Resizing and saving processed images
"""

import os
import cv2
import numpy as np


# Path configuration
im_path = 'Database'        # Raw input images directory
gt_path = 'Database_gt'     # Ground truth images directory
new_im_path = 'data/im'     # Processed images output directory
new_gt_path = 'data/gt'     # Processed ground truth output directory

if not(os.path.exists('data')):
    os.mkdir('data')
if not(os.path.exists(new_im_path)):
    os.mkdir(new_im_path)
if not(os.path.exists(new_gt_path)):
    os.mkdir(new_gt_path)


try:
    """
    Main data processing loop:
    1. Recursively walks through directory structure
    2. Processes each image/ground truth pair
    3. Converts ground truth to binary mask
    4. Saves processed versions
    """
    bpath=os.getcwd()
    gt_path_full=os.path.join(bpath,gt_path)
    d1=os.listdir(gt_path_full)
    len_d1=len(d1)
    count=0
    im_count=0
    for i1 in range(len_d1):
        path_branch_2=os.path.join(bpath,gt_path,d1[i1])
        d2 = os.listdir(path_branch_2)
        len_d2 = len(d2)
        for i2 in range(len_d2):
            path_branch_3 = os.path.join(path_branch_2, d2[i2])
            d3 = os.listdir(path_branch_3)
            len_d3 = len(d3)
            for i3 in range(len_d3):
                path_branch_4 = os.path.join(path_branch_3, d3[i3])
                img_brach_4 = os.path.join(bpath,im_path, d1[i1], d2[i2], d3[i3])
                d4 = os.listdir(path_branch_4)
                len_d4 = len(d4)
                for i4 in range(len_d4):
                    gt_filename = os.path.join(path_branch_4, d4[i4])
                    im_filename = os.path.join(img_brach_4, d4[i4])
                    f1 = os.path.splitext(gt_filename)
                    # count += 1
                    # Process only JPG/PNG images
                    if (f1[-1].lower() == '.jpg') | (f1[-1].lower() == '.png'):
                        # Read and resize images
                        gt_im=cv2.imread(gt_filename)
                        orig_im = cv2.imread(im_filename)
                        gt_im2 = cv2.resize(gt_im, (128,128),interpolation=cv2.INTER_NEAREST)
                        orig_im2 = cv2.resize(orig_im, (128, 128), interpolation=cv2.INTER_NEAREST)
                        # Convert ground truth to binary mask using specific color values
                        # (R=0, G=242, B=255) indicates positive class
                        tmp1 = gt_im2[:, :, 0] == 0    # Red channel check
                        tmp2 = gt_im2[:, :, 1] == 242  # Green channel check
                        tmp3 = gt_im2[:, :, 2] == 255  # Blue channel check
                        gt_im_binary = (tmp1 & tmp2) & tmp3  # Combine checks
                        gt_im_binary.astype(np.float)
                        gt_im_binary=gt_im_binary*255
                        print('im_count=%d, count=%d'%(im_count,count))
                        im_count+=1

                        if count>100:
                            raise StopIteration
                        if np.count_nonzero(gt_im_binary)>0:
                            im_name_new='%d.png'%(count)
                            cv2.imwrite(os.path.join(new_im_path, im_name_new), orig_im2)
                            cv2.imwrite(os.path.join(new_gt_path, im_name_new), gt_im_binary)
                            count+=1
except:
    print('finished')


