# Saves 3D .nii.gz as 2D png, lossless.
# INPUT: dtype(raw) = uint8, dtype(seg) = float64
# OUTPUT: dtype(raw, seg) = uint8 (unsigned char, aka 'ubyte' is alias for uint8 in numpy)
# Last updated: Jan 20, 2022 by charissa

import numpy as np
import glob
import os
import nibabel as nib
import cv2
import argparse
from skimage.exposure import rescale_intensity

# RAW FILES
def convert_raw_volumes(raw_files, out_path_raw):
    for i in range(0, len(raw_files)):
        im = nib.load(raw_files[i])
        im2 = np.asarray(im.dataobj)
        fname = str.split(os.path.split(raw_files[i])[1], '.nii.gz')[0]
        out_path = out_path_raw

        for j in range(0, im2.shape[2]):
            if j <= 9:
                fout = fname + '_slice00' + str(j) + '.png'
            elif 9 < j < 100:
                fout = fname + '_slice0' + str(j) + '.png'
            else:
                fout = fname + '_slice' + str(j) + '.png'
            fout2 = os.path.join(out_path, fout)
            im3 = rescale_intensity(im2[:, :, j], in_range='image', out_range='uint8')
            cv2.imwrite(fout2, im3, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f'[+] Slice saved: {fout2}', end='\r')

# SEG FILES
def convert_seg_volumes(seg_files, out_path_seg):
    for m in range(0, len(seg_files)):
        im = nib.load(seg_files[m])
        im2 = np.asarray(im.dataobj)
        fname = str.split(os.path.split(seg_files[m])[1], '.nii.gz')[0]
        out_path = out_path_seg

        for n in range(0, im2.shape[2]):
            if n <= 9:
                fout = fname + '_slice00' + str(n) + '.png'
            elif 9 < n < 100:
                fout = fname + '_slice0' + str(n) + '.png'
            else:
                fout = fname + '_slice' + str(n) + '.png'
            fout2 = os.path.join(out_path, fout)
            im3 = rescale_intensity(im2[:, :, n], in_range='image', out_range='uint8')
            cv2.imwrite(fout2, im3, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(f'[+] Slice saved: {fout2}', end='\r')

def main():
    if False:
        root_dir = '/home/charissa/Documents/minivess_files/'
        raw_dir = '/home/charissa/Documents/minivess_files/20211203_all_raw_uint16_065535/'
        seg_dir = '/home/charissa/Documents/minivess_files/20211203_all_seg_cleaned_uint8_01/'
        raw_files = sorted(glob.glob(raw_dir + '*.nii.gz'))
        seg_files = sorted(glob.glob(seg_dir + '*.nii.gz'))

        out_path_raw = os.path.join(root_dir, 'raw2/')
        out_path_seg = os.path.join(root_dir, 'seg2/')

        if not os.path.exists(out_path_raw):
            os.mkdir(out_path_raw)
        if not os.path.exists(out_path_seg):
            os.mkdir(out_path_seg)

        print('Converting {} raw files and {} segmented 3D NifTi files to 2D png format.'.format(len(raw_files), len(seg_files)))

        convert_raw_volumes(raw_files, out_path_raw)
        convert_seg_volumes(seg_files, out_path_seg)

    else:
        parser = argparse.ArgumentParser(
            description='Image conversion from 3D NifTI (.nii.gz) to 2D png (.png) format.')
        parser.add_argument('root_dir', type=str, default='/home/charissa/Documents/minivess_files/',
                            help='root directory containing directories of your raw and segmented 3D NifTi volumes')
        parser.add_argument('raw_dir', type=str, default='/home/charissa/Documents/minivess_files/20211203_all_raw_uint16_065535/',
                            help='root directory containing directories of your raw 3D NifTi volumes')
        parser.add_argument('seg_dir', type=str,
                            default='/home/charissa/Documents/minivess_files/20211203_all_seg_cleaned_uint8_01/',
                            help='root directory containing directories of your raw 3D NifTi volumes')

        args = parser.parse_args()

        root_dir = args.root_dir
        raw_dir = args.raw_dir
        seg_dir = args.seg_dir
        raw_files = sorted(glob.glob(raw_dir + '*.nii.gz'))
        seg_files = sorted(glob.glob(seg_dir + '*.nii.gz'))

        out_path_raw = os.path.join(root_dir, 'raw2/')
        out_path_seg = os.path.join(root_dir, 'seg2/')

        if not os.path.exists(out_path_raw):
            os.mkdir(out_path_raw)
        if not os.path.exists(out_path_seg):
            os.mkdir(out_path_seg)

        print('Converting {} raw files and {} segmented 3D NifTi files to 2D png format.'.format(len(raw_files),
                                                                                                 len(seg_files)))
        convert_raw_volumes(raw_files, out_path_raw)
        convert_seg_volumes(seg_files, out_path_seg)


if __name__ == "__main__":
    main()
