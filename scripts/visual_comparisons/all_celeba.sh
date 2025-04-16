export NIF_SETUP="0_65bpp"
export WEBP_SETUP="40"
export STRUMPLER_SETUP="24"
./scripts/visual_comparisons/celeba.sh 183679 40 40 20 20

export WEBP_SETUP="60"
./scripts/visual_comparisons/celeba.sh 186986 40 40 15 160

export WEBP_SETUP="30"
./scripts/visual_comparisons/celeba.sh 189985 40 40 100 160