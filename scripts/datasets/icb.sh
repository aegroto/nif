FOLDER=test_images/icb/
rm -rf $FOLDER
mkdir -p $FOLDER

wget http://imagecompression.info/test_images/rgb8bit.zip
mv rgb8bit.zip $FOLDER
cd $FOLDER
unzip rgb8bit.zip
rm readme.txt rgb8bit.zip

magick mogrify -format png *.ppm
rm *.ppm