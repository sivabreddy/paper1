from tkinter import filedialog
import cv2, numpy as np
import matplotlib.pyplot as plt
from Main.Pre_processing import *


def Select_Roi(med_im):
    # Select ROI
    check_image = med_im
    r, c,_ = np.asarray((check_image.shape)) // 2
    roi = check_image[r - r + 10:r + r - 20, c - c + 20:c + c - 20]  # ROI Extraction
    return roi

############### Input Image ###############
file_path = filedialog.askopenfilename(initialdir="Main\data\im")
path_components = file_path.split('/')
org_path = path_components[0]+'/'+path_components[1]+'/'+path_components[2]+'/'+path_components[3]+'/'+path_components[4]+'/'+path_components[5]+'/gt/'+path_components[7]
image = cv2.imread(file_path)  # input image
org = cv2.imread(org_path)
ff = file_path
img = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
org_img = cv2.resize(org, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
plt.title("Input Image")
plt.imshow(image, cmap='gray')
plt.show()

######### ROI Extraction ########
roi = Select_Roi(img)  # ROI Extraction
plt.title("ROI")
plt.imshow(roi, cmap='gray')
plt.show()
######### T2FCS ########
t2fcs = T2FCS(roi)
plt.title("T2FCS Filter")
plt.imshow(t2fcs, cmap='gray')
plt.show()
############ Segnet Segmentation ##############
seg = segment(t2fcs,org_img)
plt.title("Segnet Segmentation")
plt.imshow(seg, cmap='gray')
plt.show()
print("\nDone.")