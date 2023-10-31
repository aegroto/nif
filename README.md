# NIF: Neural Imaging Format [ACMMM '23]
[**[Paper]**](https://dl.acm.org/doi/pdf/10.1145/3581783.3613834)
[**[Project Page]**](https://iplab.dmi.unict.it/nif/)
[**[Full results]**](https://cutt.ly/nif-mm23-results) 
[**[Poster]**](https://iplab.dmi.unict.it/nif/poster.pdf)

This repository contains the source code used in experimental evaluations for the paper "NIF: A Fast Implicit Image Compression with Bottleneck Layers and Modulated Sinusoidal Activations".

# Environment setup
To create a virtual environment and to install necessary dependencies use:

```
python3 -m venv .env
source .env/bin/activate
pip3 install -r requirements.txt 
```

The ```source .env/bin/activate``` command should be always given at the beginning of a shell session, before executing any script, to enable the virtual environment with all necessary dependencies.

# Reproducing experiments
## Single image
To encode a single image run the ```encode.py``` scripts:

```
python3 encode.py <CONFIGURATION_PATH> <INPUT_IMAGE> <COMPRESSED_IMAGE>
```

For example, to encode the image \#3 from the Kodak dataset located at "test_images/kodak/3.png" and save it at "kodim03.nif" run:
```
python3 encode.py configurations/nif/kodak/120.yaml test_images/kodak/3.png kodim03.nif
```

Analogously, you can decode a compressed image using the ```decode.py``` script:

```
python3 decode.py <CONFIGURATION_PATH> <COMPRESSED_IMAGE> <DECOMPRESSED_IMAGE> 
```

To decompress the image encoded in the example above, use:
```
python3 decode.py configurations/nif/kodak/120.yaml kodim03.nif kodim03_decoded.png
```

Input files should be encoded in lossless PNG format.
Output file names end with ".nif" by convention, but this is not forced by code.

## Dataset encoding
To reproduce the experiments discussed in the paper, run the corresponding script in the folder "scripts/datasets/". These scripts will generate a bash file named "schedule.sh", filled with the sequence of commands that will encode a whole dataset. Logs will be saved under "logs/" and compressed files, along decompressed images and statistics will be saved under "results/".

For instance, to generate a "schedule.sh" to reproduce experiments on Kodak use:

```
./scripts/datasets/kodak.sh
```

Then execute it with ```./schedule.sh```.

If you want to perform a full encode-decode experiment on a single image and to export stats, you can use the "experiment.sh" script as follows:

```
./experiment.sh <CONFIGURATION_PATH> <INPUT_IMAGE> <OUTPUT_FOLDER>
```

The compressed and then decoded images, along with stats saved in JSON, will be exported in the indicated output folder.

## Downloading datasets
The kodak dataset is included in this repository, along with compressed ".nif" files that can be decoded using this software.

The sample from CelebA is the same used by Strumpler et al [2022]. To download it, follow the instructions at https://github.com/YannickStruempler/inr_based_compression/#datasets.

The ICB dataset can be downloaded at http://imagecompression.info/test_images/rgb8bit.zip. It is necessary to convert it to PNG, for this conversion we have used [ImageMagick](imagemagick.org):

```
    magick <input_file>.ppm <output_file>.png
```
