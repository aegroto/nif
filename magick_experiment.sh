date 

FILE_PATH=$1
RESULTS_ROOT=$2
FORMAT=$3
QUALITY=$4

WIDTH=$(identify -format "%w" $FILE_PATH)
HEIGHT=$(identify -format "%h" $FILE_PATH)

mkdir -p $RESULTS_ROOT

COMPRESSED_PATH=$RESULTS_ROOT/compressed.$FORMAT
STATS_PATH=$RESULTS_ROOT/stats.json

magick convert -quality $QUALITY -depth 8 -size ${WIDTH}x${HEIGHT} $FILE_PATH $COMPRESSED_PATH

python3 filewise_export_stats.py \
    $FILE_PATH \
    $COMPRESSED_PATH \
    $STATS_PATH \
    $COMPRESSED_PATH

date
