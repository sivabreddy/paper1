"""
Image Preprocessing Pipeline
---------------------------
This module handles the complete image processing workflow including:
- ROI extraction
- T2FCS filtering (Two-Threshold Fuzzy Contrast Stretching)
- Segmentation using SegNet
- Data augmentation
- Feature extraction
"""

import glob, cv2, os
import re
from Main import Proposed_SegNet, Augmentation
import numpy as np


# Output directory paths for processed images
new_im_path = 'Output/roi'       # ROI extracted images
new_fil_path = 'Output/t2fcs'    # T2FCS filtered images
new_seg_path = 'Output/segmented' # Segmented images
new_aug1_path = 'Output/rotation' # Rotated augmented images
new_aug2_path = 'Output/cropping' # Cropped augmented images

if not(os.path.exists('Output')):
    os.mkdir('Output')
if not(os.path.exists(new_im_path)):
    os.mkdir(new_im_path)

if not(os.path.exists(new_fil_path)):
    os.mkdir(new_fil_path)

if not(os.path.exists(new_seg_path)):
    os.mkdir(new_seg_path)
if not(os.path.exists(new_aug1_path)):
    os.mkdir(new_aug1_path)
if not(os.path.exists(new_aug2_path)):
    os.mkdir(new_aug2_path)

def T2FCS(m5):
    """
    Two-Threshold Fuzzy Contrast Stretching (T2FCS) filter
    Enhances image contrast using fuzzy logic and neighborhood processing
    
    Args:
        m5: Input BGR image (numpy array)
    
    Returns:
        Filtered image with enhanced contrast (numpy array)
    """
    def T2FCS_Filtering(image, row, col):
        currentElement = 0
        left = 0
        right = 0
        top = 0
        bottom = 0
        topLeft = 0
        topRight = 0
        bottomLeft = 0
        bottomRight = 0
        counter = 1
        currentElement = image[row][col]

        if not col - 1 < 0:
            left = image[row][col - 1]
            counter += 1
        if not col + 1 > width - 1:
            right = image[row][col + 1]
            counter += 1
        if not row - 1 < 0:
            top = image[row - 1][col]
            counter += 1
        if not row + 1 > height - 1:
            bottom = image[row + 1][col]
            counter += 1

        if not row - 1 < 0 and not col - 1 < 0:
            topLeft = image[row - 1][col - 1]
            counter += 1
        if not row - 1 < 0 and not col + 1 > width - 1:
            topRight = image[row - 1][col + 1]
            counter += 1
        if not row + 1 > height - 1 and not col - 1 < 0:
            bottomLeft = image[row + 1][col - 1]
            counter += 1
        if not row + 1 > height - 1 and not col + 1 > width - 1:
            bottomRight = image[row + 1][col + 1]
            counter += 1
        total = int(currentElement) + int(left) + int(right) + int(top) + int(bottom) + int(topLeft) + int(
            topRight) + int(bottomLeft) + int(bottomRight)
        avg = total / counter

        meau = avg
        n = 4

        Thr = 10
        T1 = np.arange(Thr - 2, Thr + 2)
        T2 = np.arange(Thr - 4, Thr + 4)
        T3 = np.arange(Thr - 8, Thr + 8)
        Fs = np.random.uniform(n - 1, n)

        # Case1#
        if meau in T1:
            I_x_y = currentElement
            # to find Y: absolute function of window based on neighbour pixel #
            Y = abs(I_x_y - np.mean(avg))
            # initially the value of za set to 0
            Za = 0
            Za = Za + I_x_y * Y
            # rounding process #
            Za = Za / 8

            # first value choosen as D
            D = Za
            if D > 10:
                Fij = 1 - (D - 1) / 4
            else:
                Fij = 1

            Inew = currentElement * Fij

        # Case2#
        elif meau in T2:
            I_x_y = currentElement
            # to find Y: absolute function of window based on neighbour pixel #
            Y = abs(I_x_y - np.mean(avg))
            # initially the value of za set to 0
            Za = 0
            Za = Za + I_x_y * Y
            # rounding process #
            Za = Za / 8

            # first value choosen as D
            D = Za
            if D > 10:
                Fij = 1 - (D - 1) / 4
            else:
                Fij = 1

            # new function generated as Fs
            n1 = 4
            Fs = np.sum(meau) / n1
            Inew = currentElement * (Fs / Fs)

        elif meau in T3:
            Inew = currentElement
        else:
            Inew = avg

        return Inew

    m5_tst = cv2.cvtColor(m5, cv2.COLOR_BGR2GRAY)
    height, width = m5.shape[0], m5.shape[1]
    img_t2fcs = np.zeros((height, width, 3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_t2fcs[i, j, 0] = T2FCS_Filtering(m5[:, :, 0], i, j)
            img_t2fcs[i, j, 1] = T2FCS_Filtering(m5[:, :, 1], i, j)
            img_t2fcs[i, j, 2] = T2FCS_Filtering(m5[:, :, 2], i, j)
    return img_t2fcs

def mark_seg_in_orgim(input, seg):
    """
    Marks segmentation results on original image for visualization
    
    Args:
        input: Original input image
        seg: Segmentation mask (binary image)
    
    Returns:
        Image with segmentation marked (numpy array)
    """
    input = input[:, :, 0]
    input = cv2.resize(input, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    input = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
    seg = cv2.resize(seg, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            if (seg[i][j] == 255):
                input[i + 1][j] = (0, 255, 255)
                input[i][j + 2] = (0, 255, 255)
    return input


def segment(input_im, org):
    """
    Performs image segmentation using SegNet model
    
    Args:
        input_im: Input image to segment
        org: Original image for reference
    
    Returns:
        Segmented image (numpy array)
    """
    input_im = cv2.resize(input_im, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    org = cv2.resize(org, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    org = cv2.cvtColor(org, cv2.COLOR_BGR2GRAY)
    # Segmentation
    seg = Proposed_SegNet.Segnet_Segmentation(input_im,org)
    seg = mark_seg_in_orgim(input_im,seg)
    return seg

def Select_Roi(med_im, count):
    """
    Selects Region of Interest (ROI) from input image
    
    Args:
        med_im: Input image
        count: Counter for saving output files
    
    Returns:
        Extracted ROI (numpy array)
    """
    # Select ROI
    check_image = med_im
    r, c,_ = np.asarray((check_image.shape)) // 2
    roi = check_image[r - r + 10:r + r - 20, c - c + 20:c + c - 20]  # ROI Extraction
    cv2.imwrite('Main/Output/roi/roi_' + str(count) + '.png', roi)
    return roi

def augment():
    file_path = 'Output/segmented//*.png'
    files = glob.glob(file_path)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    g_path = 'data/gt//*.png'
    files_g = glob.glob(g_path)
    files_g.sort(key=lambda f: int(re.sub('\D', '', f)))
    count = 0
    Feat = []
    label = []
    for i in range(count,len(files)):
        print(files[i])
        input = cv2.imread(files[i])
        input = cv2.resize(input, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        seg = cv2.imread(files_g[i])
        seg = cv2.resize(seg, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        ########## STEP5: Augmentation ###########
        rotate, crop = Augmentation.Augmentation(input)
        cv2.imwrite(new_aug1_path + "//" + str(count) + '.png', rotate)
        cv2.imwrite(new_aug2_path + "//" + str(count) + '.png', crop)
        f1 = np.histogram(input[~seg], 100)
        f1 = f1[0]
        Feat.append(f1.tolist())
        label.append(0)
        f2 = np.histogram(input[seg], 100)
        f2 = f2[0]
        Feat.append(f2.tolist())
        label.append(1)
        f3=np.histogram(rotate[~seg], 100)
        f3 = f3[0]
        Feat.append(f3.tolist())
        label.append(0)
        f4 = np.histogram(rotate[seg], 100)
        f4 = f4[0]
        Feat.append(f4.tolist())
        label.append(1)
        f5 = np.histogram(crop[~seg], 100)
        f5 = f5[0]
        Feat.append(f5.tolist())
        label.append(0)
        f6 = np.histogram(crop[seg], 100)
        f6 = f6[0]
        Feat.append(f6.tolist())
        label.append(1)
        count+=1
    np.savetxt("Feat.csv",Feat,delimiter=',',fmt='%s')
    np.savetxt("Label.csv",label,delimiter=',',fmt='%s')
    return Feat

def pre_process():
    # read database and extract features #
    file_path='data/im//*.png'
    gt_path = 'data/gt/*.png'
    files_gt = glob.glob(gt_path)
    files_gt.sort(key=lambda f: int(re.sub('\D', '', f)))
    files = glob.glob(file_path)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    count=0
    for i in range(count,len(files)):
        print(files[i])
        ########## STEP1: Read Database ###########
        input = cv2.imread(files[i])
        org_im = cv2.imread(files_gt[i])
        input_im = cv2.resize(input, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        org_im = cv2.resize(org_im, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
        ######### STEP2: ROI Extraction ########
        roi = Select_Roi(input_im,count)  # ROI Extraction
        cv2.imwrite(new_im_path+"//" + str(count) + '.png', roi)
        ######### STEP3: Pre-processing #########
        t2fcs = T2FCS(roi)
        cv2.imwrite(new_fil_path+"//" + str(count) + '.png', t2fcs)
        ######### STEP4: SEGNET Segmentation #########
        seg = segment(t2fcs,org_im)
        cv2.imwrite(new_seg_path + "//" + str(count) + '.png', seg)
        count +=1

def processing():
    pre_process() # ROI,Preprocessing,T2FCS,Segmentation
    augment() # augmentation

def process():
    print("\n >> ROI Extraction..")
    print("\n >> Preprocessing..")
    print("\n >> T2FCS Filter..")
    print("\n >> Segnet Segmentation..")
    #processing()
