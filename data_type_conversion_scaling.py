# Adapted from Henrik's,
# Shared on Sept 17, 2021.
# Unfortunately, uint8 is not supported by nifti (see nifti1 header).
# Update: Sept 25, 2021. Added scaling to halve x,y dim.
# Last updated March 30th, 2022

import glob
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
import os
import csv
import nrrd
import sys
#import oiffile as oif  # reads oib only

# reads oir and oib
import javabridge
import bioformats
javabridge.start_vm(class_path=bioformats.JARS)


export_path_raw = "/home/charissa/minivess/data/20220331_out_nifti_raw/"
if not os.path.isdir(export_path_raw):
    os.mkdir(export_path_raw)

export_path_seg = "/home/charissa/minivess/data/20220331_all_seg_cleaned_01_temp/"
if not os.path.isdir(export_path_seg):
    os.mkdir(export_path_seg)

# set directories and file lists
raw_path = '/home/charissa/minivess/data/20220327_out_nifti_raw/'
raw_files = sorted(glob.glob(raw_path+'*.nii.gz'))
seg_path = '/home/charissa/minivess/data/20220327_all_seg_cleaned_01/'
seg_files = sorted(glob.glob(seg_path+'*.nii.gz'))
oibr_path = '/home/charissa/minivess/data/out_nifti_oib/'
oib_files = glob.glob(oibr_path+'*.oib')
oir_files = glob.glob(oibr_path+'*.oir')
oibr_files = sorted(oib_files + oir_files)
nrrd_path = '/home/charissa/minivess/data/20220331_slicer_seg/'
nrrd_files = sorted(glob.glob(nrrd_path + '*.nrrd'))

def find_oibr_raw(fname2):
    # find oib/oir file for metadata
    ffind_oib = oibr_path + fname2 + '.oib'
    ffind_oir = oibr_path + fname2 + '.oir'
    if ffind_oib in oibr_files:
        ind = oibr_files.index(ffind_oib)
        oibr_raw = bioformats.load_image(oibr_files[ind])  # np array
        oibr_meta = bioformats.get_omexml_metadata(oibr_files[ind])  # oir/oib meta
        return(oibr_raw, oibr_meta)
    elif ffind_oir in oibr_files:
        ind = oibr_files.index(ffind_oir)
        oibr_raw = bioformats.load_image(oibr_files[ind])  # np array
        oibr_meta = bioformats.get_omexml_metadata(oibr_files[ind])  # oir/oib meta
        return (oibr_raw, oibr_meta)
    else:
        print('no oibr raw found for file:', fname2)

#%%  FOR SEG .nrrd -->  FILES
for j in range(len(nrrd_files)):
    nrrd_im, nrrd_header = nrrd.read(nrrd_files[j])

    fname = os.path.split(nrrd_files[j])[1]
    fname2 = str.split(fname,'.nii.gz')[0][:-2]  # get ext and '_y' off
    if 't-0' in fname2:
        fname2 = fname2[:-5]
    fout = export_path_seg + fname2 + '_y.nii.gz'

    oibr_raw, oibr_meta = find_oibr_raw(fname2)

    # for correct scaling in nifti1 header,
    # get physical sizes of xyz dims
    size_x_string = 'PhysicalSizeX'
    size_y_string = 'PhysicalSizeY'
    size_z_string = 'PhysicalSizeZ'
    #dim_x_string = 'SizeX'  # no space in this for Nifti1 header
    #dim_y_string = 'SizeY'
    #dim_z_string = 'SizeZ'
    size_x_loc = oibr_meta.find(size_x_string) + 15  #[15:20]
    size_y_loc = oibr_meta.find(size_y_string) + 15  #[15:20]
    size_z_loc = oibr_meta.find(size_z_string) + 15  #[15:18]
    size_x = float(oibr_meta[size_x_loc:size_x_loc+5])
    size_y = float(oibr_meta[size_y_loc:size_y_loc+5])
    size_z = float(oibr_meta[size_z_loc:size_z_loc+3])
    #print(size_x, size_y, size_z)

    affine = np.eye(4)
    affine[0,0] = size_x  #oib.rex
    affine[1,1] = size_y  #oib.rey
    affine[2,2] = size_z  #oib.res_Z

    print(nrrd_im.dtype, nrrd_im.max(), nrrd_im.min())

    # SCALE IMAGE
    #dim = (256,256)
    #im_scale = cv2.resize(im2, dim, interpolation=cv2.INTER_NEAREST)
    
    new_image = nib.Nifti1Image(nrrd_im, affine=affine)
    #new_image = nib.Nifti1Image((im_scale).astype(target_dtype), affine=affine)
    #new_image.header['scl_slope'] = 1.0/scale_value
    new_image.header['scl_slope'] = 1.0
    new_image.header['scl_inter'] = 0.0
    new_image.header['dim'][0] = len(nrrd_im.shape)
    new_image.header['dim'][3] = nrrd_im.shape[-1]
    new_image.header['dim'][4] = 1.
    new_image.header['pixdim'][1] = size_x  
    new_image.header['pixdim'][2] = size_y 
    new_image.header['pixdim'][3] = size_z 
    new_image.header['cal_min'] = 0.
    new_image.header['cal_max'] = 1.
    new_image.header['bitpix'] = 8  # 16 for uint16, 8 for uint8
    #new_image.header['xyzt_units'] = [3]  # 3 for microns
    print(new_image.header)
    
    #print(new_image.dataobj.dtype, new_image.dataobj.max(), new_image.dataobj.min(), new_image.shape)

    nib.save(new_image, fout)


#%% FOR RAW/SEG .nii.gz FILES
def export_niftis(files, flag):
    for i in range(len(files)):
        nib_im = nib.load(files[i])
    
        fname = os.path.split(files[i])[1]
        fname2 = str.split(fname,'.nii.gz')[0]  # get ext off
        if 't-0' in fname2:
            fname2 = fname2[:-5]
        print(fname2)
            
        if flag == 'raw':
            fout = export_path_raw + fname2 + '.nii.gz'
        elif flag == 'seg':
            fout = export_path_seg + fname2 + '_y.nii.gz'
        else:
            print('flag error')
        print(fout)
    
        oibr_raw, oibr_meta = find_oibr_raw(fname2)
    
        # for correct scaling in nifti1 header,
        # get physical sizes of xyz dims
        size_x_string = 'PhysicalSizeX'
        size_y_string = 'PhysicalSizeY'
        size_z_string = 'PhysicalSizeZ'
        #dim_x_string = 'SizeX'  # no space in this for Nifti1 header
        #dim_y_string = 'SizeY'
        #dim_z_string = 'SizeZ'
        size_x_loc = oibr_meta.find(size_x_string) + 15  #[15:20]
        size_y_loc = oibr_meta.find(size_y_string) + 15  #[15:20]
        size_z_loc = oibr_meta.find(size_z_string) + 15  #[15:18]
        size_x = float(oibr_meta[size_x_loc:size_x_loc+5])
        size_y = float(oibr_meta[size_y_loc:size_y_loc+5])
        size_z = float(oibr_meta[size_z_loc:size_z_loc+3])
        print(size_x, size_y, size_z)
        if size_x == float(0) or size_y == float(0) or size_z == float(0) :
            print('error with size_x, size_y, size_z, at i=',i, fname2)
            sys.exit()
    
        affine = np.eye(4)
        affine[0,0] = size_x  #oib.rex
        affine[1,1] = size_y  #oib.rey
        affine[2,2] = size_z  #oib.res_Z
    
        ## SET TARGET DTYPE
        #if img has already the oib dtype you want, then just pass it like that
        #otherwise I do sth like this
        #dtype = im.get_data_dtype()  # FOR NII.GZ IMAGES. keep uint16 as target dtype
        if flag == 'raw':
            target_dtype = np.uint16
        elif flag == 'seg':
            target_dtype = np.uint8
        else:
            print('FLAG ERROR')
            sys.exit()
        im = np.array(nib_im.dataobj)  
        im2 = im.astype(target_dtype)
        print('im2 dtype:', im2.dtype)
        
        ## ENSURE DIM SIZE IS 3 (X,Y,Z)
        if len(im2.shape) == 4:
            im2 = im2[:,:,:,0]
        elif len(im2.shape) == 3:
            pass
        else:
            print('error in im.shape:', im.shape)
            sys.exit()
        print('im2.shape:', im2.shape)
            
        #new_image = nib.Nifti1Image((im_scale).astype(target_dtype), affine=affine)
        new_image = nib.Nifti1Image(im2, affine=affine)
        new_image.header['scl_slope'] = 1.0 #1.0/scale_value
        new_image.header['scl_inter'] = 0.0
        new_image.header['dim'][0] = len(im2.shape)
        new_image.header['dim'][3] = im2.shape[-1]
        new_image.header['dim'][4] = 1.
        new_image.header['pixdim'][1] = size_x  # pixdim[0] must be 1 or -1
        new_image.header['pixdim'][2] = size_y  # pixdim[0] must be 1 or -1
        new_image.header['pixdim'][3] = size_z  # pixdim[0] must be 1 or -1
        
        ## HEADER FIELDS THAT DEPEND ON RAW/SEG
        if flag == 'raw':
            new_image.header['bitpix'] = 16  # 16 for uint16, 8 for uint8, https://nipy.org/nibabel/nifti_images.html  
            new_image.header['cal_min'] = 0
            new_image.header['cal_max'] = 1000
        elif flag == 'seg':
            new_image.header['bitpix'] = 8  # 16 for uint16, 8 for uint8
            new_image.header['cal_min'] = 0
            new_image.header['cal_max'] = 1
        else:
            print('FLAG ERROR part 2')
            sys.exit()
            
        #new_image.header['xyzt_units'] = [3]  # 3 for microns
        print(new_image.header)
        print(i)
        
        nyan = input('Continue?')
        if nyan == ':':
            pass
        else:
            sys.exit()
        
        nib.save(new_image, fout)
    
        #print(i, fname, ', slope:', new_image.dataobj.slope, ', intercept:', new_image.dataobj.inter, new_image.dataobj.min(), new_image.dataobj.max())  # REAL NEWS

export_niftis(raw_files, flag='raw')
export_niftis(seg_files, flag='seg')

#%% for files whose x,y dims were changed by slicer,
# apply binary closing and median filter to last uncorrupted segmentation
# then converted to nii.gz using export_niftis, line-by-line
from scipy.ndimage.morphology import binary_closing
from scipy.ndimage import median_filter

# if the last uncorrupted seg is .nii.gz
meow = nib.load('/home/charissa/minivess/data/20220317_all_seg_cleaned_uint8_01/CP-20170305-87573-day0-son2-700mVpp_t-00_y.nii.gz')
# if the last uncorrupted seg is .nrrd
meow, head = nrrd.read('/home/charissa/minivess/data/20220323_slicer_seg/CP-20180310_mouse_TN4_nobubbs3_t-00_y.nii.gz.nii.seg.nrrd')
meow.shape
meow2 = np.asarray(meow.dataobj)
plt.imshow(meow2[:,:,0], cmap='gray')
plt.show()
plt.imshow(meow2[:,:,-1], cmap='gray')
meow_new = np.zeros(meow2.shape)
for i in range(0, meow2.shape[2]):
    meow_closed = binary_closing(meow2[:,:,i])
    meow_filter = median_filter(meow_closed, size=3)
    meow_closed = binary_closing(meow2[:,:,i])
    meow_filter = median_filter(meow_closed, size=5)
    meow_filter2 = meow_filter.astype(np.uint8)
    meow_new[:,:,i] = meow_filter2

plt.imshow(meow_new[:,:,0], cmap='gray')
plt.show()
plt.imshow(meow_new[:,:,-1], cmap='gray')
meow_new = meow_new.astype(np.uint8)
np.unique(meow_new)

#%% for seg files that have values outside of [0,1]
# best way to check is by np.unique(array), which gives you dtype as well
# then converted to nii.gz using export_niftis, line-by-line
raw = nib.load('/home/charissa/minivess/data/20220330_out_nifti_raw/CP-20201007_cw_v_XYZ_scan02.nii.gz')
seg = nib.load('/home/charissa/minivess/data/20220330_all_seg_cleaned_01/CP-20201007_cw_v_XYZ_scan02_y.nii.gz')
raw_im = np.asarray(raw.dataobj)
seg_im = np.asarray(seg.dataobj)
np.unique(seg_im)

plt.imshow(raw_im[:,:,0])
plt.imshow(seg_im[:,:,0])
plt.imshow(seg_im2[:,:,0])
plt.imshow(raw_im[:,:,8])
plt.imshow(seg_im[:,:,8])
plt.imshow(seg_im2[:,:,-1])

seg_im2 = np.copy(seg_im)
seg_im2[seg_im >= 1] = 1
np.unique(seg_im2)
nib_im = seg_im2
im2 = seg_im2
fname2 = 'CP-20201007_cw_v_XYZ_scan02'

#%% sanity check that im and seg shapes match

raw_path = '/home/charissa/minivess/data/20220331_out_nifti_raw/'
seg_path = '/home/charissa/minivess/data/20220330_all_seg_cleaned_01/'

raw_f2 = sorted(glob.glob(raw_path + '*.nii.gz'))
print(len(raw_f2))
seg_f2 = sorted(glob.glob(seg_path + '*.nii.gz'))
assert(len(raw_f2) == len(seg_f2))

summary = []

for r in range(25,len(raw_f2)):
    fname = os.path.split(raw_f2[r])[1]
    fname_nii_im = fname
    fname_nii_seg = fname[:-7] + '_y.nii.gz'
    #fname_nii_im =  'CP-20170203-900-prettyXYZ.nii.gz'
    #fname_nii_seg = 'CP-20170203-900-pretty-XYZ_y.nii.gz'
    #fname_nii_im = 'CP-20160319-CRND8-86748-TR-anat1.nii.gz'
    #fname_nii_seg = 'CP-20160319-CRND8-86748-TR-anat1-2_y.nii.gz'
    
    im_dir = raw_path + fname_nii_im
    seg_dir = seg_path + fname_nii_seg
    
    print(im_dir)
    print(seg_dir)
    
    im = nib.load(im_dir)
    seg = nib.load(seg_dir)
    im_d = im.dataobj
    seg_d = seg.dataobj
    
    print('im:', im_d.shape)
    print('seg:', seg_d.shape)
    print(r)
    
    assert(im_d.shape == seg_d.shape)
    assert(im_d.dtype.name == 'uint16')
    assert(seg_d.dtype.name == 'uint8')
    assert(np.unique(seg_d)[0] == 0)
    assert(np.unique(seg_d)[-1] == 1)
    
    #assert(im.header['scl_slope']) == 0
    #assert(im.header['scl_inter']) == 1
    #assert(seg.header['scl_slope']) == 0
    #assert(seg.header['scl_inter']) == 1

    assert(im.header['dim'][0] == seg.header['dim'][0])
    assert(im.header['dim'][3] == seg.header['dim'][3])
    assert(im.header['dim'][4]) == 1.
    assert(seg.header['dim'][4]) == 1.
    assert(im.header['pixdim'][1] == seg.header['pixdim'][1])
    assert(im.header['pixdim'][2] == seg.header['pixdim'][2])
    assert(im.header['pixdim'][3] == seg.header['pixdim'][3])
    print(im.header['pixdim'][1], im.header['pixdim'][2], im.header['pixdim'][3])
    
    assert(im.header['bitpix']) == 16
    assert(im.header['cal_min']) == 0
    assert(im.header['cal_max']) == 1000
    
    assert(seg.header['bitpix']) == 8
    assert(seg.header['cal_min']) == 0
    assert(seg.header['cal_max']) == 1
    
    
    # alias name
    if r < 9:
        alias_raw_fname = 'mv'+'0'+str(r+1)+'.nii.gz'
        alias_seg_fname = 'mv'+'0'+str(r+1)+'_y.nii.gz'
    else:
        alias_raw_fname = 'mv'+str(r+1)+'.nii.gz'
        alias_seg_fname = 'mv'+str(r+1)+'_y.nii.gz'    
    
    summary.append([fname_nii_im, alias_raw_fname, im_d.dtype.name, 
                    fname_nii_seg, alias_seg_fname, seg_d.dtype.name, 
                    im_d.shape[0], im_d.shape[1], im_d.shape[2],
                    im.header['pixdim'][1],im.header['pixdim'][2],im.header['pixdim'][3]])

with open('20220331_minivess_summary2.csv', 'w') as f:
    write = csv.writer(f)
    fields = ['raw file name', 'raw alias file name', 'raw dtype',
              'seg file name', 'seg alias file name', 'seg dtype',
              'x size', 'y size', 'z size',
              'x pix size', 'y pix size', 'z pix size']
    write.writerow(fields)
    for row in range(len(summary)):
        write.writerow(summary[row])
 

