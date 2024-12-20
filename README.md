# MiniVess: A dataset of rodent cerebrovasculature from in vivo multiphoton fluorescence microscopy imaging.

We provide the code for image preprocessing, conversion, and segmentation of the paper [A dataset of rodent cerebrovasculature from in vivo multiphoton fluorescence microscopy imaging](https://www.biorxiv.org/content/10.1101/2022.07.19.500542v1.abstract) by Charissa Poon, Petteri Teikari, Muhammad Febrian Rachmadi, Henrik Skibbe, and Kullervo Hynynen.

The data is stored in an EBRAINS repository: [https://search.kg.ebrains.eu/instances/bf268b89-1420-476b-b428-b85a913eb523](https://search.kg.ebrains.eu/instances/bf268b89-1420-476b-b428-b85a913eb523)

## Abstract
We present MiniVess, the first annotated dataset of rodent cerebrovasculature, acquired using two-photon fluorescence microscopy. MiniVess consists of 70 3D image volumes with segmented ground truths. Segmentations were created using traditional image processing operations, a U-Net, and manual proofreading. Code for image preprocessing steps and the U-Net are provided. Supervised machine learning methods have been widely used for automated image processing of biomedical images. While much emphasis has been placed on the development of new network architectures and loss functions, there has been an increased emphasis on the need for publicly available annotated, or segmented, datasets. Annotated datasets are necessary during model training and validation. In particular, datasets that are collected from different labs are necessary to test the generalizability of models.  We hope this dataset will be helpful in testing the reliability of machine learning tools for analyzing biomedical images.

<h3>Description of files:</h3>

1. convert_oibr_to_nifti_ometif_publish2.py </br>
Converts 3D or 4D Olympus (.oir, .oib) files to 3D NifTi volumes, with metadata encoded in NifTi-1 header format.</br>
Example file to test code: [json_alias_x.nii.gz]

2. convert_3d_to_2d_mycode.py </br>
Converts 3D NifTi volumes to 2D png images.
We chose to use a 2D instead of a 3D UNet due to memory constraints. 
The 2D UNet receives as input 2D images, but 3D volumes are more useful to work with for scientists.
So raw 3D NifTi images must first be converted to 2D images, fed into the UNet, and then returned to 3D NifTi format.
Example file to test code: [json_alias_x.nii.gz]

3. convert_2dpng_to_3dniftiometif.py </br>
Converts 2D png images to 3D NifTi volumes. 
Specifically, converts 2D output/segmented png images from the 2D UNet, back to 3D NifTi format. 
Requires corresponding raw 3D NifTi volumes to ensure that the segmented 3D NifTi volumes have correct metadata in the header.

4. minivess_diceloss-working-2dunet_2_publish.ipynb </br>
2D UNet using MONAI framework, heavily based on their spleen segmentation tutorial.
Input: use raw 2D png images, or output from convert_3d_to_2d_mycode.py

5. image_preprocessing_example.py </br>
Example code of image preprocessing steps described in the manuscript.

<h3>Sample command line runs:</h3>

1. convert_oibr_to_nifti_ometif_publish2.py</br>
python convert_oibr_to_nifti_ometif_publish2.py --data_dir '/path_to_directory_of_oib_and_oir_files/'

2. convert_3d_to_2d_mycode.py</br>
python convert_3d_to_2d_mycode.py '/path_to_root_directory/' '/path_to_directory_of_raw_3d_nifti_volumes/' '/path_to_directory_of_raw_3d_nifti_volumes/'

   outputs are saved in directories in '/path_to_root_directory/' 

3. convert_2dpng_to_3dniftiometif.py</br>
python convert_2dpng_to_3dniftiometif.py '/path_to_root_directory/' '/path_to_directory_of_segmented_2d_png_images/' '/path_to_directory_of_raw_3d_nifti_volumes/'

