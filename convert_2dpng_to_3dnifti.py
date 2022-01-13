# Converts 2d png files to 3d nifti volumes for segmented outputs from UNet.
# Last updated Jan 13, 2022 by charissa.

import os
import glob
from collections import Counter
import numpy as np
import cv2
import tifffile
import nibabel as nib
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt

# Set directories.
png_path = '/home/charissa/Documents/minivess_files/20211102_2dunet_outputs/'
raw_path = '/home/charissa/Documents/minivess_files/20211203_all_raw_uint16_065535/'
all_png = sorted(glob.glob(png_path + '*.png'))
all_raw = sorted(glob.glob(raw_path + '*.nii.gz'))
out_nifti_path = '/home/charissa/Documents/minivess_files/20211107_seg_uint8_2d_3dnifti/'
out_ometif_path = '/home/charissa/Documents/minivess_files/20211107_seg_uint8_2d_3dometif/'

if not os.path.exists(out_nifti_path):
    os.makedirs(out_nifti_path)
if not os.path.exists(out_ometif_path):
    os.makedirs(out_ometif_path)

# Subfunctions.
def find_raw_nifti(fname_base_raw):
    temp_path = raw_path + fname_base_raw + '.nii.gz'
    if temp_path in all_raw:
        raw_nifti = nib.load(temp_path)
        raw_header = raw_nifti.header
        raw_affine = raw_nifti.affine
    else:
        print('No raw NifTi volume found at:', temp_path)
    return raw_header, raw_affine

def save_as_nifti(volume, fname_base_raw, out_nifti_path):
    volume2 = rescale_intensity(volume, in_range='image', out_range=(0,1)).astype('uint8')

    # Use the same NifTi header as for raw volumes to preserve pixel dims etc., but save segmented volume as uint8.
    nifti_header, nifti_affine = find_raw_nifti(fname_base_raw)
    nifti_header['data_type'] = 2  # unsigned 8-bit char
    nifti_header['bitpix'] = 8
    nifti_header['cal_min'] = 0
    nifti_header['cal_max'] = 1000

    new_nifti = nib.Nifti1Image(volume2, nifti_affine, nifti_header)

    fout = out_nifti_path + fname_base_raw + '.nii.gz'
    nib.save(new_nifti, fout)

def save_as_ometif(volume, fname_base_raw, out_ometif_path):
    volume2 = rescale_intensity(volume, in_range='image', out_range=(0,1)).astype('uint8')

    # Move axes for ometif from (512,512,Z) --> (Z,512,512) or whatever XY.
    volume3 = np.moveaxis(volume2, [0,1,2], [1,2,0])

    fout = out_ometif_path + fname_base_raw + '.ometif'
    tifffile.imwrite(fout, volume3)

def main():
    # Get unique file names for each stack and respective number of Z slices.
    unq_list = []
    for n in range(0, (len(all_png))):
        this_name = all_png[n][:-13]
        unq_list.append(this_name)
    unq_fnames_keys = np.unique(unq_list)
    unq_fnames_dict = dict(Counter(unq_list))

    save_nifti = input('Save files as NifTi? (True/False)')
    save_ometif = input('Save files as OMETIF? (True/False)')

    sslice_counter = 0
    for f in range(len(unq_fnames_keys)):
        fname = unq_fnames_keys[f]
        num_slices = unq_fnames_dict[fname]
        for s in range(num_slices):
            sslice = np.array(cv2.imread(all_png[s + sslice_counter]))
            if s == 0:  # first slice
                volume = np.zeros((sslice.shape[0], sslice.shape[1], num_slices))
            sslice_gray = cv2.cvtColor(sslice, cv2.COLOR_BGR2GRAY)
            volume[:,:,s] = sslice_gray
            sslice_counter += 1
        fname_base_raw = os.path.split(fname)[1]

        if save_nifti:
            save_as_nifti(volume, fname_base_raw, out_nifti_path)
        if save_ometif:
            save_as_ometif(volume, fname_base_raw, out_ometif_path)

        print('Exported {}.'.format(fname_base_raw))

if __name__ == "__main__":
    main()