# > schedule.sh

LOSS=$1

for config in $(find configurations/ablations/losses/kodak/$LOSS/*.yaml)
do
    config_file=$(basename $config)
    config_id="${config_file%.yaml}"
    for i in {1..24}
    do
        log_file="logs/${config_id}_$i.txt"
        echo "./experiment.sh $config test_images/kodak/$i.png results/nif/losses/kodak/$LOSS/$config_id/$i > $log_file 2>&1" >> schedule.sh
    done
done
