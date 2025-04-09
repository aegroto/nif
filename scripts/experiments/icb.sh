> schedule.sh

for config in $(find configurations/nif/icb/*.yaml)
do
    config_file=$(basename $config)
    config_id="${config_file%.yaml}"
    for file in test_images/icb/*.png
    do
        basename=$(basename $file)
        i=${basename%.*}
        echo "uv run bash experiment.sh $config test_images/icb/$i.png results/nif/icb/$config_id/$i" >> schedule.sh
    done
done

