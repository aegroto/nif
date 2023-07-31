> schedule.sh

for config in $(find configurations/ablations/celeba_equal_layers/*.yaml)
do
    config_file=$(basename $config)
    config_id="${config_file%.yaml}"
    for file in test_images/celeba/*.png
    do
        basename=$(basename $file)
        i=${basename%.*}

        log_file="logs/${config_id}_$i.txt"
        echo "./experiment.sh $config test_images/celeba/$i.png results/nif/celeba_equal_layers/$config_id/$i > $log_file 2>&1" >> schedule.sh
    done
done
