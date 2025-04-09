DATASET="set5_hr"

> schedule.sh

for config in $(find configurations/speed/*.yaml)
do
    config_file=$(basename $config)
    config_id="${config_file%.yaml}"
    for file in test_images/${DATASET}/*.png
    do
        basename=$(basename $file)
        i=${basename%.*}

        log_file="logs/${config_id}_$i.txt"
        echo "./speed_experiment.sh $config $file results/nif/${DATASET}_speed/$config_id/$i > $log_file 2>&1" >> schedule.sh
    done
done

# echo "python3 calculate_summary.py stats.json results/summaries/${DATASET}_speed/nif.json results/nif/${DATASET}_speed/" >> schedule.sh
# echo "python3 plot_results.py results/summaries/${DATASET}_speed/ results.png psnr" >> schedule.sh
