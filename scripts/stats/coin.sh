date 

FILE_PATH=$1
RESULTS_ROOT=$2

COMPRESSED_PATH=$RESULTS_ROOT/best_model.pt
DECODED_PATH=$RESULTS_ROOT/decoded.png
STATS_PATH=$RESULTS_ROOT/stats.json

python3 coin_export_stats.py \
    $FILE_PATH \
    $DECODED_PATH \
    $STATS_PATH \
    $COMPRESSED_PATH

date
