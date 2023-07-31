DATASET=$1
SETUP=$2

for file in test_images/$DATASET/*.png;
do
    image_filename=$(basename $file)
    image_id=${image_filename%.*}
    echo "Calculating stats for $image_id in setup $SETUP..."
    ./scripts/stats/coin.sh $file results/coin/$DATASET/$SETUP/$image_id/
done
