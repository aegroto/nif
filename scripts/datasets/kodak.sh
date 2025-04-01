> schedule.sh

reprozip_root="results/.reprozip/nif/kodak/"
echo "rm -r $reprozip_root" >> schedule.sh
echo "mkdir -p $reprozip_root" >> schedule.sh

for config in $(find configurations/nif/kodak/*.yaml)
do
    config_file=$(basename $config)
    config_id="${config_file%.yaml}"
    for i in {1..24}
    do
        results_root="results/nif/kodak/$config_id/$i"
        echo "uv run reprozip trace --continue -d $reprozip_root bash experiment.sh $config test_images/kodak/$i.png $results_root" >> schedule.sh
    done
done
