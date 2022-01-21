# minivess

Data repository for MiniVess: A dataset of rodent cerebrovasculature from \textit{in vivo} multiphoton fluorescence microscopy imaging.

Description of files:
1. convert_oibr_to_nifti_ometif_publish2.py
Converts 3D or 4D Olympus (.oir, .oib) files to 3D NifTi volumes, with metadata encoded in NifTi-1 header format.

2. convert_3d_to_2d_mycode.py
Converts 3D NifTi volumes to 2D png images.
We chose to use a 2D instead of a 3D UNet due to memory constraints. 
The 2D UNet receives as input 2D images, but 3D volumes are more useful to work with for scientists.
So raw 3D NifTi images must first be converted to 2D images, fed into the UNet, and then returned to 3D NifTi format.

3. convert_2dpng_to_3dniftiometif.py
Converts 2D png images to 3D NifTi volumes. 
Specifically, converts 2D output/segmented png images from the 2D UNet, back to 3D NifTi format. 
Requires corresponding raw 3D NifTi volumes to ensure that the segmented 3D NifTi volumes have correct metadata in the header.

4. minivess_diceloss-working-2dunet_2_publish.ipynb
2D UNet using MONAI framework, heavily based on their spleen segmentation tutorial.


Sample command line runs:

1. convert_oibr_to_nifti_ometif_publish2.py
python convert_oibr_to_nifti_ometif_publish2.py --data_dir '/path_to_directory_of_oib_and_oir_files/'

2. convert_3d_to_2d_mycode.py
python convert_3d_to_2d_mycode.py '/path_to_root_directory/' '/path_to_directory_of_raw_3d_nifti_volumes/' '/path_to_directory_of_raw_3d_nifti_volumes/'

outputs are saved in directories in '/path_to_root_directory/' 

3. convert_2dpng_to_3dniftiometif.py