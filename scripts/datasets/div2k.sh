> schedule.sh

# for config in $(find configurations/.tuning/*.toml)
# for config in "configurations/default.toml"
for config in configurations/.tuning/1200.toml
do
    config_file=$(basename $config)
    config_id="${config_file%.toml}"
    for file in test_images/div2k/*.png
    do
        basename=$(basename $file)
        i=${basename%.*}

        log_file="logs/${config_id}_$i.txt"
        echo "./experiment.sh $config test_images/div2k/$i.png results/nif/div2k/$config_id/$i > $log_file 2>&1" >> schedule.sh
    done
done

# echo "python3 calculate_summary.py stats.json results/summaries/div2k/nif.json results/nif/div2k/" >> schedule.sh
# echo "python3 plot_results.py results/summaries/div2k/ results.png psnr" >> schedule.sh
