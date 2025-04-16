> schedule.sh

for config in $(find configurations/ablations/kodak_equal_layers/*.yaml)
do
    config_file=$(basename $config)
    config_id="${config_file%.yaml}"
    for i in {1..24}
    do
        echo "uv run bash experiment.sh $config test_images/kodak/$i.png results/nif/kodak_equal_layers/$config_id/$i" >> schedule.sh
    done
done
