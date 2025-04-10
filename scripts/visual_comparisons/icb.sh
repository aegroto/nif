IMAGE_ID=$1
CROP_W=$2
CROP_H=$3
CROP_X=$4
CROP_Y=$5

CROP_PARAMS="${CROP_W}x${CROP_H}+${CROP_X}+${CROP_Y}"
CROP_RECTANGLE="${CROP_X},${CROP_Y} $((${CROP_W}+${CROP_X})),$((${CROP_H}+${CROP_Y}))"

DRAW_PARAMS="-stroke white -strokewidth 3 -fill transparent" 

RESULTS_PATH="visual_comparisons/icb_$IMAGE_ID"
STATS_PATH="visual_comparisons/icb_$IMAGE_ID/stats/"

rm -r $RESULTS_PATH
# mkdir -p $RESULTS_PATH/full
mkdir -p $RESULTS_PATH/crop
mkdir -p $STATS_PATH

# magick "test_images/icb/$IMAGE_ID.png" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/ground_truth.png"

# magick "results/nif/icb/$NIF_SETUP/$IMAGE_ID/decoded.png" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/nif.png"
cp "results/nif/icb/$NIF_SETUP/$IMAGE_ID/stats.json" "$STATS_PATH/nif.json"

# magick "results/jpeg/icb/$JPEG_SETUP/$IMAGE_ID/compressed.jpeg" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/jpeg.png"
cp "results/jpeg/icb/$JPEG_SETUP/$IMAGE_ID/stats.json" "$STATS_PATH/jpeg.json"

# magick "results/webp/icb/$WEBP_SETUP/$IMAGE_ID/compressed.webp" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/webp.png"
cp "results/webp/icb/$WEBP_SETUP/$IMAGE_ID/stats.json" "$STATS_PATH/webp.json"

# magick "results/bpg/icb/$BPG_SETUP/$IMAGE_ID/decoded.png" $DRAW_PARAMS -draw "rectangle $CROP_RECTANGLE" "$RESULTS_PATH/full/bpg.png"
cp "results/bpg/icb/$BPG_SETUP/$IMAGE_ID/stats.json" "$STATS_PATH/bpg.json"

magick "test_images/icb/$IMAGE_ID.png" -crop $CROP_PARAMS "$RESULTS_PATH/crop/ground_truth.png"
magick "results/nif/icb/$NIF_SETUP/$IMAGE_ID/decoded.png" -crop $CROP_PARAMS "$RESULTS_PATH/crop/nif.png"
magick "results/jpeg/icb/$JPEG_SETUP/$IMAGE_ID/compressed.jpeg" -crop $CROP_PARAMS "$RESULTS_PATH/crop/jpeg.png"
magick "results/webp/icb/$WEBP_SETUP/$IMAGE_ID/compressed.webp" -crop $CROP_PARAMS "$RESULTS_PATH/crop/webp.png"
magick "results/bpg/icb/$BPG_SETUP/$IMAGE_ID/decoded.png" -crop $CROP_PARAMS "$RESULTS_PATH/crop/bpg.png"