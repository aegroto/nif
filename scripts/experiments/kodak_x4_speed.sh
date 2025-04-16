> schedule.sh

for config in $(find configurations/nif/kodak_x4_speed/*.yaml)
do
    config_file=$(basename $config)
    config_id="${config_file%.toml}"
    for i in {1..24}
    do
        echo "uv run bash experiment.sh $config test_images/kodak/$i.png results/nif/kodak_x4_speed/$config_id/$i" >> schedule.sh
    done
done
