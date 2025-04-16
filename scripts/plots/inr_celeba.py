import common as plot

plot.init(1, 1)

plot.add_results_path("../../results/summaries/celeba/inr/*.json")
plot.plot_metric(0, 0, "", "psnr", "PSNR (dB)", None, None, 1, 1, bpp_key="state_bpp")
plot.clear_results()

plot.set_dump_path("../../plots/inr_celeba.pdf")
plot.save(2.5, 2.5)
