CONFIGURATION_PATH=$1
FILE_PATH=$2
RESULTS_ROOT=$3

rm -r $RESULTS_ROOT
mkdir -p $RESULTS_ROOT

COMPRESSED_PATH=$RESULTS_ROOT/compressed.nif
DECODED_PATH=$RESULTS_ROOT/decoded.png
STATS_PATH=$RESULTS_ROOT/stats.json

LOGS_ROOT=$RESULTS_ROOT/.logs/
mkdir -p $LOGS_ROOT

uv run -m encode $CONFIGURATION_PATH $FILE_PATH $COMPRESSED_PATH > $LOGS_ROOT/encode.log 2>&1 
uv run -m decode $CONFIGURATION_PATH $COMPRESSED_PATH $DECODED_PATH > $LOGS_ROOT/decode.log 2>&1 

uv run -m filewise_export_stats \
    $FILE_PATH \
    $DECODED_PATH \
    $STATS_PATH \
    $COMPRESSED_PATH > $LOGS_ROOT/stats.log 2>&1 
