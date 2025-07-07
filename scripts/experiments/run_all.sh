# Set the CELEBA_DOWNLOAD_FOLDER environment variable beforehand

# Experiments
./scripts/experiments/kodak.sh
./schedule.sh
./scripts/datasets/celeba.sh $CELEBA_DOWNLOAD_FOLDER
./scripts/experiments/celeba.sh
./schedule.sh
./scripts/datasets/icb.sh
./scripts/experiments/icb.sh
./schedule.sh

# Plotting
cd scripts/plots/
uv run -m inr_kodak
uv run -m inr_celeba
uv run -m traditional_kodak
