#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 14:10:30 2023

@author: charissa

Example code for image preprocessing steps described in the MiniVess manuscript.
"""

## Import libraries
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import nrrd
import skimage

from scipy.ndimage.morphology import binary_closing
from scipy.ndimage import median_filter


#%% 

## 1. Load the image volume.
# nifti images:
nib_obj = nib.load('mv00_y.nii.gz')
nib_im = np.asarray(nib_obj.dataobj)
# nrrd images: (eg, images saved after manual corrections using Slicer)
nrrd_obj, head = nrrd.read('your_nrrd_file.nrrd')

## 2. Visualize original image volume.
plt.imshow(nib_im[:,:,0], cmap='gray')
plt.show()
plt.imshow(nib_im[:,:,-1], cmap='gray')

## 3. Create a new array of the same shape to save the new image array in.
nib_im_new = np.zeros(nib_im.shape)



########## Image processing examples below ###########
### NB: nib_im_new is overwritten in the code below, but processed volumes can 
### be saved as new variables as well.                                 

## 4a. Image preprocessing example 1: Apply binary closing and median filters 
# twice to the image volume to fill holes and smooth segmentations, respectively.
for i in range(0, nib_im.shape[2]):
    nib_im_closed = binary_closing(nib_im[:,:,i])
    nib_im_filter = median_filter(nib_im_closed, size=3)
    nib_im_closed = binary_closing(nib_im[:,:,i])
    nib_im_filter = median_filter(nib_im_closed, size=5)
    
    nib_im_new[:,:,i] = nib_im_filter
    
## 4b. Image preprocessing example 2: Contrast limited adaptive histogram 
# equalization, followed by gaussian smoothing.
for j in range(0, nib_im.shape[2]):
    nib_im_new[:, :, j] = skimage.exposure.equalize_adapthist(nib_im[:, :, j])
nib_im_new = skimage.filters.gaussian(nib_im_new, sigma=.2, multichannel=False)

## 4c. Image preprocessing example 3: Binarize the image volume (if it is not 
# already binary)
#fig, ax = try_all_threshold(nib_im_new[:,:,14], figsize=(10, 8), verbose=False)
thresh = skimage.filters.threshold_yen(nib_im_new)
nib_im_new = nib_im_new > thresh



## 5. Visualize first and last slices of the volume.
plt.imshow(nib_im_new[:,:,0], cmap='gray')
plt.show()
plt.imshow(nib_im_new[:,:,-1], cmap='gray')
plt.show()

## 6. Save new image volume as a nifti.
new_header = nib_obj.header
new_affine = nib_obj.affine
#print(new_header)

# Define directory to save new image volume.
new_dir = 'path_to_save_new_image_volume/data/'
# If the directory does not exist, create it.
os.makedirs(new_dir, exist_ok=True)

new_name = new_dir + 'new_image_file_name' + '.nii.gz'
new_nib = nib.Nifti1Image(nib_im_new, new_affine, new_header)
nib.save(new_nib, new_name)
