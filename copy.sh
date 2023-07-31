for conf in "0_4bpp" "32.yaml" "60.yaml" "120.yaml"
do
    dest="results/nif/celeba_compare/$conf/"
    mkdir -p $dest
    for i in "182664" "185277" "190719" "200044" "202322"
    do
        cp -r "results/nif/celeba_state_old/$conf/$i/" $dest
    done
done
