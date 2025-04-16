import common as plot

plot.init(1, 2)

plot.add_results_path(f"../../results/summaries/highres/*.json")
plot.plot_metric(0, 0, "", "psnr", "PSNR (dB)", [0.0, 0.4], [28.0, 36.0], 1, 1)
plot.plot_metric(0, 1, "", "ms-ssim", "MS-SSIM", [0.0, 0.4], [5.5, 16.0], 1, 1, normalizer = plot.MS_SSIM_NORM)
plot.clear_results()

plot.set_dump_path(f"../../plots/traditional_icb.pdf")
plot.save(2.5, 6.5)
