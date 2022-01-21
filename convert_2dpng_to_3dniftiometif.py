# Converts 2d png files to 3d nifti volumes for segmented outputs from UNet.
# Last updated Jan 20, 2022 by charissa.

import os
import glob
from collections import Counter
import numpy as np
import cv2
import tifffile
import nibabel as nib
from skimage.exposure import rescale_intensity
import argparse

# Subfunctions.
def find_raw_nifti(raw_path, fname_base_raw, all_raw):
    temp_path = raw_path + fname_base_raw + '.nii.gz'
    if temp_path in all_raw:
        raw_nifti = nib.load(temp_path)
        raw_header = raw_nifti.header
        raw_affine = raw_nifti.affine
    else:
        print('No raw NifTi volume found at:', temp_path)
    return raw_header, raw_affine

def save_as_nifti(volume, fname_base_raw, out_nifti_path, raw_path, all_raw):
    volume2 = rescale_intensity(volume, in_range='image', out_range=(0,1)).astype('uint8')

    # Use the same NifTi header as for raw volumes to preserve pixel dims etc., but save segmented volume as uint8.
    nifti_header, nifti_affine = find_raw_nifti(raw_path, fname_base_raw, all_raw)
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
    if False:
        # Set input directories.
        png_path = input('Path to segmented 2d png files: ')
        while not os.path.exists(png_path):
            print('That png path does not exist.')
            png_path = input('Path to segmented 2d png files: ')
        png_path = os.path.join(png_path,'')  # adds trailing fwd slash
        raw_path = input('Path to the raw 3D nifti files: ')
        while not os.path.exists(raw_path):
            print('That raw path does not exist.')
            raw_path = input('Path to the raw 3D nifti files: ')
        raw_path = os.path.join(raw_path,'')  # adds trailing fwd slash

        save_nifti = input('Save files as NifTi? (True/False)  ')
        save_ometif = input('Save files as OMETIF? (True/False)  ')

        root_dir = os.path.split(png_path)[0]

    else:
        parser = argparse.ArgumentParser(
            description='Image conversion from 2D png (.png) to 3D NifTI (.nii.gz) format.')
        parser.add_argument('root_dir', type=str, help='root directory where output directories will be stored')
        parser.add_argument('png_path', type=str, help='directory containing segmented 2d png images')root_dir
        parser.add_argument('raw_path', type=str, help='directory containing raw 3d NifTi volumes')
        parser.add_argument('-save_nifti', '--save_nifti', default=True, action='store_true',
                            help='Export as NifTI format, commonly used for radiological imaging and in DL frameworks like MONAI')
        parser.add_argument('-save_ometif', '--save_ometif', default=True, action='store_true',
                            help='Export as OME-TIFF format, commonly used for microscopy images.')

        args = parser.parse_args()

        print('root_dir:', args.root_dir)
        print('png_path:', args.png_path)
        print('raw_path:', args.raw_path)

        root_dir = os.path.join(args.root_dir,'')
        png_path = os.path.join(args.png_path,'')
        raw_path = os.path.join(args.raw_path,'')
        if args.save_nifti:
            save_nifti = True
        else:
            save_nifti = False
        if args.save_ometif:
            save_ometif = True
        else:
            save_ometif = False

        print('argparse else statement done')

    # Set output directories
    if save_nifti:
        out_nifti_path = os.path.join(root_dir, 'out_nifti2/')
        if not os.path.exists(out_nifti_path):
            os.makedirs(out_nifti_path)
    if save_ometif:
        out_ometif_path = os.path.join(root_dir, 'out_ometif2/')
        if not os.path.exists(out_ometif_path):
            os.makedirs(out_ometif_path)

    all_png = sorted(glob.glob(png_path + '*.png'))
    all_raw = sorted(glob.glob(raw_path + '*.nii.gz'))

    print('{} all_png files and {} all_raw files'.format(len(all_png), len(all_raw)))

    # Get unique file names for each stack and respective number of Z slices.
    unq_list = []
    for n in range(0, (len(all_png))):
        this_name = all_png[n][:-13]
        unq_list.append(this_name)
    unq_fnames_keys = np.unique(unq_list)
    unq_fnames_dict = dict(Counter(unq_list))

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
            save_as_nifti(volume, fname_base_raw, out_nifti_path, raw_path, all_raw)
        if save_ometif:
            save_as_ometif(volume, fname_base_raw, out_ometif_path)

        print('Exported {}.'.format(fname_base_raw))

if __name__ == "__main__":
    main()