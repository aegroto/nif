import math
import common as plot

plot.init(1, 2)

plot.add_results_path(f"../../results/summaries/kodak/traditional/*.json")
plot.add_results_path(f"../../results/summaries/kodak/autoencoder/*.json")
plot.add_results_path(f"../../results/summaries/kodak/inr/strumpler_basic_8bit.json")
plot.add_results_path(f"../../results/summaries/kodak/inr/nif.json")
plot.plot_metric(0, 0, "", "psnr", "PSNR (dB)", None, [22.0, 42.0], 1, 1)
plot.plot_metric(0, 1, "", "ms-ssim", "MS-SSIM", None, None, 1, 1, normalizer = plot.MS_SSIM_NORM)
plot.clear_results()

plot.set_dump_path(f"../../plots/traditional_kodak.pdf")
plot.save(3.0, 7.5)
