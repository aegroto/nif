CODEC=$1
DATASET=$2
SETUP=$3

for file in test_images/$DATASET/*.png;
do
    image_filename=$(basename $file)
    image_id=${image_filename%.*}
    echo "Calculating stats for $image_id in setup $SETUP..."
    ./scripts/stats/stats_only.sh $file results/$CODEC/$DATASET/$SETUP/$image_id/
done
