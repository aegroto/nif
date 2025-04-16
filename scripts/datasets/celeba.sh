#!/bin/bash
CELEBA_DOWNLOAD_FOLDER=$1
ROOT_FOLDER=$PWD

# Unzip the full dataset
cd $CELEBA_DOWNLOAD_FOLDER
for archive in *.zip
do
    unzip $archive
done

# Extract the 100 images used in the experiments
cd $ROOT_FOLDER

mkdir -p test_images/celeba/

7z e \
    "${CELEBA_DOWNLOAD_FOLDER}/img_align_celeba_png.7z/img_align_celeba_png.7z.001" \
    $(cat test_images/.celeba_images.txt) \
    -otest_images/celeba/

rm -r $CELEBA_DOWNLOAD_FOLDER/img_align_celeba_png.7z

