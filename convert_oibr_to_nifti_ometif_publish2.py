#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 10:05:31 2022
Last updated on Thurs Jan 20
@author: charissa
"""

## Converts oir and oib files to nifti and ometif using the tifffile library.
# TODO! nifti datatype is 32 bit and i cannot figure out why
# wip: oir_oib_to_nifti.py

## Libraries

import os
import glob
import numpy as np
from skimage.exposure import rescale_intensity

import tifffile
import jpype
import pims
import pylab
import nibabel as nib
import argparse

## Subfunctions

def convert_a_dir_of_oir_and_oib(data_dir, export_as_nifti, export_as_ometif):
    oir_files = sorted(glob.glob(os.path.join(data_dir, '*.oir')))
    oib_files = sorted(glob.glob(os.path.join(data_dir, '*.oib')))
    oib_oir_files = oir_files + oib_files
    print('Found a total of {} .oir files + {} .oib files = {} files to convert'.format(len(oir_files),len(oib_files),len(oib_oir_files)))

    # Create output directories for nifti and ometif files.
    if export_as_nifti:
        nifti_dir = os.path.join(os.path.split(data_dir)[0],'out_nifti2')
        if not os.path.exists(nifti_dir):
            os.makedirs(nifti_dir)
        print('Nifti output will be saved in directory: ', nifti_dir)

    if export_as_ometif:
        ometif_dir = os.path.join(os.path.split(data_dir)[0],'out_ometif2')
        if not os.path.exists(ometif_dir):
            os.makedirs(ometif_dir)
        print('OME-TIF output will be saved in directory: ', ometif_dir)

    for i, oib_oir_filepath in enumerate(oib_oir_files):
        fname_only = os.path.split(oib_oir_filepath)[1]
        print('\nConverting #{}/{}: "{}"'.format(i + 1, len(oib_oir_files), fname_only))

        # Actually convert the single .oir or .oib to .nii.gz
        convert_single_file(oib_oir_filepath, nifti_dir, ometif_dir,
                            export_as_nifti = export_as_nifti,
                            export_as_ometif = export_as_ometif)

def convert_single_file(oib_oir_filepath, nifti_dir, ometif_dir,
                            export_as_nifti, export_as_ometif):
    #pims.bioformats.download_jar(version='6.5')
    reader = pims.Bioformats(oib_oir_filepath)
    fname_only = os.path.split(reader.filename)[1]
    print('Image dimensions of {}: {}'.format(fname_only, reader.sizes))

    # For multichannel images, choose 1 channel to export as a stack.
    # NB. When c==1, there is no 'c' in reader.sizes.
    if 'c' in reader.sizes:
        img_to_export = choose_channel(reader)
    else:
        img_to_export = np.array(reader)

    # Get the metadata.
    img_metadata = reader.metadata

    # For multi-T image stacks, ask user if each T-stack should be split before exporting.
    # If yes, split T, swap axes, and export.
    # XYZT files treated separately here as the separate T-stacks are a list, not a single file.
    if 't' in reader.sizes and 'z' in reader.sizes:
        print('Image dimensions (T,Z,Y,X):', img_to_export.shape)
        print('Full XYZT stack will be exported.')
        t_split = input('Split image stack into separate T-stacks? (y/n)')
        if t_split == 'y' or t_split == 'Y':
            t_stacks = np.split(img_to_export, img_to_export.shape[0], axis=0)
            # For each t_stack (list), swap axes, export and save as nifti.
            for t in range(len(t_stacks)):
                t_stacks_moved_nifti = np.moveaxis(t_stacks[t][0], [0, 1, 2], [-1, -2, -3])
                print('\nExporting T-stack #{} with shape: {}'.format(t, t_stacks_moved_nifti.shape))
                if export_as_nifti:
                    fname_new = 't-' + str(t) + '_' + fname_only
                    convert_to_nifti(t_stacks_moved_nifti, img_metadata, fname_new, nifti_dir)
                if export_as_ometif:
                    fname_new = 't-' + str(t) + '_' + fname_only
                    t_stacks_moved_ometif = t_stacks[t][0]
                    convert_to_ometif(t_stacks_moved_ometif, img_metadata, fname_new, ometif_dir)

    # For TZYX (no t-stack separation) and ZYX, TYX.
    # NB. Move axes JUST before export to nifti/ometif.
    print('\nExporting full image stack of shape: {}'.format(img_to_export.shape))
    img_to_export_moved_ometif = img_to_export  # Z,512,512
    if len(img_to_export.shape) == 4:  # TZYX (no t-stack separation)
        img_to_export_moved_nifti = np.moveaxis(img_to_export, [0, 1, 2, 3], [-1, -2, -3, -4])
        print('\nThe shape of the array with the shifted axis is: {}'.format(img_to_export_moved_nifti.shape))
    if len(img_to_export.shape) == 3:  # ZYX, TYX.
        img_to_export_moved_nifti = np.moveaxis(img_to_export, [0, 1, 2], [-1, -2, -3])
        print('\nThe shape of the array with the shifted axis is: {}'.format(img_to_export_moved_nifti.shape))

    if export_as_nifti:
        convert_to_nifti(img_to_export_moved_nifti, img_metadata, fname_only, nifti_dir)
    if export_as_ometif:
        convert_to_ometif(img_to_export_moved_ometif, img_metadata, fname_only, ometif_dir)

def choose_channel(reader):
    # If image stack has multiple channels, choose 1 channel to export.

    if 't' in reader.sizes and 'z' in reader.sizes:  # CTZYX (fix T==0?)
        reader.bundle_axes = 'ctzyx'
        reader_array = np.array(reader)

        print('Choose the Z-slice to view images. Z-slice must be between 1 and', reader.sizes['z'], '.')
        z_slice = int(input('Z-slice:'))
        while z_slice < 1 or z_slice > reader.sizes['z']:
            print('Z-slice must be between 1 and', reader.sizes['z'], '.')
            z_slice = int(input('Z-slice:'))
        z_slice = z_slice - 1  # for 0 indexing
        for c in range(reader.sizes['c']):
            pylab.subplot(1, reader.sizes['c'], c + 1)
            pylab.imshow(reader_array[0][c][0][z_slice])
            pylab.pause(1)
        print('Channels: 1 to', reader.sizes['c'])
        ch_to_export = int(input('Choose a channel to export:')) - 1
        while (ch_to_export+1) < 1 or (ch_to_export+1) > reader.sizes['c']:
            print('Channel must be between 1 and', reader.sizes['c'], '.')
            ch_to_export = int(input('Choose a channel to export:')) - 1
        img_to_export = reader_array[0][ch_to_export]

    elif not 't' in reader.sizes and 'z' in reader.sizes:  # CZYX
        reader.bundle_axes = 'czyx'
        reader_array = np.array(reader)

        print('Choose the Z-slice to view images. Z-slice must be between 1 and', reader.sizes['z'], '.')
        z_slice = int(input('Z-slice:'))
        while z_slice < 1 or z_slice > reader.sizes['z']:
            print('Z-slice must be between 1 and', reader.sizes['z'], '.')
            z_slice = int(input('Z-slice:'))
        z_slice = z_slice - 1  # for 0 indexing
        for c in range(reader.sizes['c']):
            pylab.subplot(1, reader.sizes['c'], c + 1)
            pylab.imshow(reader_array[0][c][z_slice])
            pylab.pause(1)
        print('Channels: 1 to ', reader.sizes['c'])
        ch_to_export = int(input('Choose a channel to export:')) - 1
        img_to_export = reader_array[0][ch_to_export]

    elif 't' in reader.sizes and not 'z' in reader.sizes:  # CTYX (fix T=0)
        reader.bundle_axes = 'ctyx'
        reader_array = np.array(reader)

        for c in range(reader.sizes['c']):
            pylab.subplot(1, reader.sizes['c'], c + 1)
            pylab.imshow(reader_array[0][c][0])
            pylab.pause(1)
        print('Channels: 1 to ', reader.sizes['c'])
        ch_to_export = int(input('Choose a channel to export:')) - 1
        img_to_export = reader_array[0][ch_to_export]

    return(img_to_export)

def convert_to_nifti(img_to_export_moved_nifti, img_metadata, fname_only, nifti_dir):
    # https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
    affine_matrix = np.eye(4)
    affine_matrix[0,0] = img_metadata.PixelsPhysicalSizeX(0)
    affine_matrix[1,1] = img_metadata.PixelsPhysicalSizeY(0)
    affine_matrix[2,2] = img_metadata.PixelsPhysicalSizeZ(0)

    # transfer metadata to nifti1 header https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/dim.html
    nifti_header = nib.Nifti1Header()
    nifti_header['dim'][1] = img_metadata.PixelsSizeX(0)  # 512
    nifti_header['dim'][2] = img_metadata.PixelsSizeY(0)
    nifti_header['dim'][3] = img_metadata.PixelsSizeZ(0)
    if not img_metadata.PixelsSizeT:
        nifti_header['dim'][4] = 1
    else:
        nifti_header['dim'][4] = img_metadata.PixelsSizeT(0)
    nifti_header['pixdim'][1] = img_metadata.PixelsPhysicalSizeX(0)  # pixdim = voxel width, corresponds to xyzt_units
    nifti_header['pixdim'][2] = img_metadata.PixelsPhysicalSizeY(0)
    nifti_header['pixdim'][3] = img_metadata.PixelsPhysicalSizeZ(0)
    nifti_header['xyzt_units'] = 3  # 'NIFTI_UNITS_MICRON'
    nifti_header['scl_slope'] = 1.0
    nifti_header['scl_inter'] = 0.0
    nifti_header['data_type'] = 512  # uint16
    nifti_header['bitpix'] = 16
    nifti_header['cal_min'] = 0
    nifti_header['cal_max'] = 1000

    img_to_export_moved_nifti2 = rescale_intensity(img_to_export_moved_nifti, in_range='image', out_range='uint16')
    img_to_export_moved_nifti2.astype(np.uint16)
    nifti_obj = nib.Nifti1Image(img_to_export_moved_nifti2, affine_matrix, nifti_header)

    if '.oir' in fname_only:
        fname_out_nifti = fname_only.replace('.oir', '.nii.gz')
    else:
        fname_out_nifti = fname_only.replace('.oib', '.nii.gz')
    path_out_nifti = os.path.join(nifti_dir, fname_out_nifti)
    nib.save(nifti_obj, path_out_nifti)

def convert_to_ometif(img_to_export_ometif, img_metadata, fname_only, ometif_dir):
    if '.oir' in fname_only:
        fname_out_ometif = fname_only.replace('.oir', '.ometif')
    else:
        fname_out_ometif = fname_only.replace('.oib', '.ometif')
    path_out_ometif = os.path.join(ometif_dir, fname_out_ometif)
    img_to_export2 = rescale_intensity(img_to_export_ometif, in_range='image', out_range='uint16').astype(np.uint16)
    tifffile.imwrite(path_out_ometif, img_to_export2)


## MAIN

#if __name__ == "__main__":
def main():
    if False:
        #dir_path = os.path.dirname(os.path.realpath(__file__)) # use if DATASET is in the same directory
        dir_path = '/home/charissa/Dropbox/vessel_2PM_data_paper' #'/home/charissa/minivess/out_nifti_oib'
        data_dir = os.path.join(dir_path, 'DATASET/raw_files_oib')  #dir_path

        # Ask user whether to convert to nifti and/or ome-tif files
        export_as_nifti = input('Export as nifti files? (True/False): ')
        export_as_ometif = input('Export as ometif files? (True/False): ')

        print('Converting all the .oir and .oib files to .nii.gz found from = "{}"'.format(data_dir))

        convert_a_dir_of_oir_and_oib(data_dir, export_as_nifti, export_as_ometif)
    else:
        parser = argparse.ArgumentParser(description='Image conversion from Olympus (.oib, .oir) to NifTI (.nii.gz) and/or OME-TIFF (.ometif).')
        parser.add_argument('-data_dir', '--data_dir', type=str, default='../DATASET/raw_files_oib/',
                        help='directory containing your .oib files')
        
        #parser.add_argument('-export_as_nifti', '--export_as_nifti', action='store_true',
        parser.add_argument('export_as_nifti', action='store_true',
                        help='Export as NifTI format, commonly used for radiological imaging and in DL frameworks like MONAI')

        #parser.add_argument('-export_as_ometif', '--export_as_ometif', action='store_true',
        parser.add_argument('export_as_ometif', action='store_true',
                        help='Export as OME-TIFF format, commonly used for microscopy images.')

        args = parser.parse_args()

        export_as_nifti = args.export_as_nifti
        export_as_ometif = args.export_as_ometif

        print('Converting all the .oir and .oib files to .nii.gz found from = "{}"'.format(args.data_dir))

        convert_a_dir_of_oir_and_oib(args.data_dir, args.export_as_nifti, args.export_as_ometif)
       
if __name__ == "__main__":
    main()
    


