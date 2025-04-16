import common as plot

plot.init(1, 2)

plot.add_results_path(f"../../results/summaries/celeba/traditional/*.json")
plot.add_results_path(f"../../results/summaries/celeba/autoencoder/*.json")
plot.add_results_path(f"../../results/summaries/celeba/inr/nif.json")
plot.plot_metric(0, 0, "", "psnr", "PSNR (dB)", [0.0, 4.0], [20.0, 50.0], 1, 1)
plot.plot_metric(0, 1, "", "ms-ssim", "MS-SSIM", [0.0, 3.0], [0.92, 1.0], 1, 2)
plot.clear_results()

plot.set_dump_path(f"../../plots/traditional_celeba.pdf")
plot.save(2.5, 6.5)
