> schedule.sh

for config in $(find configurations/ablations/kodak_quantization/fit/*.yaml)
do
    config_file=$(basename $config)
    config_id="${config_file%.yaml}"
    for i in {1..24}
    do
        file_path="test_images/kodak/$i.png"
        results_path="results/nif/kodak_quantization/$config_id/$i"
        log_file="logs/${config_id}_$i.txt"

        mkdir -p $results_path
        echo "python3 fit.py $config $file_path $results_path/uncompressed.pth > $log_file 2>&1" >> schedule.sh
    done
done
