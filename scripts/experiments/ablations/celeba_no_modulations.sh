# > schedule.sh

for config in $(find configurations/ablations/celeba_no_modulations/*.yaml)
do
    config_file=$(basename $config)
    config_id="${config_file%.yaml}"
    for file in test_images/celeba/*.png
    do
        basename=$(basename $file)
        i=${basename%.*}

        echo "uv run bash experiment.sh $config test_images/celeba/$i.png results/nif/celeba_no_modulations/$config_id/$i" >> schedule.sh
    done
done
