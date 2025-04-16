import common as plot

plot.init(1, 2)

plot.add_results_path("../../results/summaries/kodak/inr/*.json")
plot.plot_metric(
    0, 0, "", "psnr", "PSNR (dB)", [0.0, 2.0], [22.5, 35.0], 1, 1, bpp_key="state_bpp"
)
plot.plot_metric(0, 1, "", "ms-ssim", "MS-SSIM", None, None, 1, 2, bpp_key="state_bpp")

plot.set_dump_path("../../plots/inr_kodak.pdf")
plot.save(2.5, 6.0)
