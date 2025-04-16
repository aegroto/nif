IMAGE_ID=$1
CROP_W=$2
CROP_H=$3
CROP_X=$4
CROP_Y=$5

CROP_PARAMS="${CROP_W}x${CROP_H}+${CROP_X}+${CROP_Y}"
CROP_RECTANGLE="${CROP_X},${CROP_Y} $((${CROP_W}+${CROP_X})),$((${CROP_H}+${CROP_Y}))"

DRAW_PARAMS="-stroke white -strokewidth 3 -fill transparent" 

RESULTS_PATH="visual_comparisons/kodak_$IMAGE_ID"
STATS_PATH="visual_comparisons/kodak_$IMAGE_ID/stats/"

rm -r $RESULTS_PATH
mkdir -p $RESULTS_PATH/full
mkdir -p $RESULTS_PATH/crop
mkdir -p $STATS_PATH

magick "test_images/kodak/$IMAGE_ID.png" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/ground_truth.png"

magick "results/nif/kodak/$NIF_SETUP/$IMAGE_ID/decoded.png" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/nif.png"
cp "results/nif/kodak/$NIF_SETUP/$IMAGE_ID/stats.json" "$STATS_PATH/nif.json"

magick "results/coin/kodak/$COIN_SETUP/$IMAGE_ID/decoded.png" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/coin.png"
cp "results/coin/kodak/$COIN_SETUP/$IMAGE_ID/stats.json" "$STATS_PATH/coin.json"

magick "results/jpeg/kodak/$JPEG_SETUP/$IMAGE_ID/compressed.jpeg" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/jpeg.png"
cp "results/jpeg/kodak/$JPEG_SETUP/$IMAGE_ID/stats.json" "$STATS_PATH/jpeg.json"

magick "results/webp/kodak/$WEBP_SETUP/$IMAGE_ID/compressed.webp" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/webp.png"
cp "results/webp/kodak/$WEBP_SETUP/$IMAGE_ID/stats.json" "$STATS_PATH/webp.json"

magick "results/strumpler/kodak/$STRUMPLER_SETUP/$IMAGE_ID/decoded.png" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/strumpler.png"
cp "results/strumpler/kodak/$STRUMPLER_SETUP/$IMAGE_ID/stats.json" "$STATS_PATH/strumpler.json"

magick "results/bpg/kodak/$BPG_SETUP/$IMAGE_ID/decoded.png" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/bpg.png"
cp "results/bpg/kodak/$BPG_SETUP/$IMAGE_ID/stats.json" "$STATS_PATH/bpg.json"

magick "results/invcompress/kodak/$INVCOMPRESS_SETUP/$IMAGE_ID/decoded.png" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/invcompress.png"
cp "results/invcompress/kodak/$INVCOMPRESS_SETUP/$IMAGE_ID/stats.json" "$STATS_PATH/invcompress.json"

magick "test_images/kodak/$IMAGE_ID.png" -crop $CROP_PARAMS "$RESULTS_PATH/crop/ground_truth.png"
magick "results/nif/kodak/$NIF_SETUP/$IMAGE_ID/decoded.png" -crop $CROP_PARAMS "$RESULTS_PATH/crop/nif.png"
magick "results/coin/kodak/$COIN_SETUP/$IMAGE_ID/decoded.png" -crop $CROP_PARAMS "$RESULTS_PATH/crop/coin.png"
magick "results/jpeg/kodak/$JPEG_SETUP/$IMAGE_ID/compressed.jpeg" -crop $CROP_PARAMS "$RESULTS_PATH/crop/jpeg.png"
magick "results/webp/kodak/$WEBP_SETUP/$IMAGE_ID/compressed.webp" -crop $CROP_PARAMS "$RESULTS_PATH/crop/webp.png"
magick "results/strumpler/kodak/$STRUMPLER_SETUP/$IMAGE_ID/decoded.png" -crop $CROP_PARAMS "$RESULTS_PATH/crop/strumpler.png"
magick "results/bpg/kodak/$BPG_SETUP/$IMAGE_ID/decoded.png" -crop $CROP_PARAMS "$RESULTS_PATH/crop/bpg.png"
magick "results/invcompress/kodak/$INVCOMPRESS_SETUP/$IMAGE_ID/decoded.png" -crop $CROP_PARAMS "$RESULTS_PATH/crop/invcompress.png"