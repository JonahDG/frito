from jax import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch
import cmasher as cmr
import ehtplot
import scienceplots
import dorito as drt
import dLux.utils as dlu

plt.style.use(["science", "bright", "no-latex"])
new_rcParams = {
    "image.cmap": "inferno",
    "font.family": "serif",
    "image.origin": "lower",
    "figure.dpi": 300,
    "font.size": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
}
plt.rcParams.update(new_rcParams)

inferno = mpl.colormaps["inferno"]
viridis = mpl.colormaps["viridis"]
seismic = mpl.colormaps["seismic"]
coolwarm = mpl.colormaps["coolwarm"]

inferno.set_bad("k", 0.5)
viridis.set_bad("k", 0.5)
seismic.set_bad("k", 0.5)
coolwarm.set_bad("k", 0.5)


def main_mosaic(
    target: str,
    filter: str,
    ois,
    optics_diam,
    deconvolution_model,
    deconvolution_result,
    bfgs_model,
    bfgs_result,
    epochs: int,
    log_dist_lr: float,
    contrast_lr: float,
    prior,
    stretch_amount: float = 0.5,
    final_loss_percent: float = 0.5,
    pixel_scale: float | None = None,
    roll_angle_degrees: float | None = None,
    diff_lim: float | None = None,
    scale: float | None = None,
    save: None | str = None,
):
    """
    Create a comprehensive mosaic plot showing deconvolution and BFGS results.

    Parameters
    ----------
    target : str
        Target name (e.g., 'PDS-70')
    filter : str
        Filter name (e.g., 'F380M')
    ois : list
        List of observation instances
    optics_diam : float
        Optics diameter
    deconvolution_model : Model
        The deconvolution model
    deconvolution_result : Result
        Result from deconvolution optimization
    bfgs_model : Model
        The BFGS-optimized model
    bfgs_result : dict
        Dictionary containing BFGS results with keys:
        'loss_before', 'loss_after', 'grad_inf_before', 'grad_inf_after'
    epochs : int
        Number of training epochs
    log_dist_lr : float
        Learning rate for log distribution
    contrast_lr : float
        Learning rate for contrast
    prior : 
        Prior value
    stretch_amount : float, optional
        Power normalization for stretched images (default: 0.5)
    final_loss_percent : float, optional
        Fraction of loss history to show in final loss plot (default: 0.5)
    pixel_scale : float, optional
        Pixel scale in arcsec. If None, computed from deconvolution_model and ois[0]
    roll_angle_degrees : float, optional
        Roll angle in degrees. If None, computed from -ois[0].parang
    diff_lim : float, optional
        Diffraction limit in arcsec. If None, computed from ois[0] and optics_diam
    scale : float, optional
        Scale parameter for plot_result. If None, not included in common params
    save : None or str, optional
        If provided, save figure to this path
    """

    fig, ax_dict = plt.subplot_mosaic(
        [
            [
                "stretched_decon",
                "stretched_decon",
                "stretched_bfgs",
                "stretched_bfgs",
                "text",
                "text",
            ],
            ["sd_cbar", "sd_cbar", "sbfgs_cbar", "sbfgs_cbar", "text", "text"],
            ["decon", "decon", "bfgs", "bfgs", "residual", "residual"],
            ["d_cbar", "d_cbar", "bfgs_cbar", "bfgs_cbar", "r_cbar", "r_cbar"],
            ["contrast", "contrast", "contrast", "latent", "latent", "latent"],
            ["loss", "loss", "loss", "fin_loss", "fin_loss", "fin_loss"],
        ],
        figsize=(18, 24),
        height_ratios=[4, 0.15, 4, 0.15, 2, 2],
    )

    fig.suptitle(f"{target} - {filter}")

    # Get distributions
    dist_pre = deconvolution_model(ois[0])
    dist_post = bfgs_model(ois[0])
    n = dist_pre.shape[0]
    pre_plot = dist_pre.at[n // 2, n // 2].set(np.nan)
    post_plot = dist_post.at[n // 2, n // 2].set(np.nan)
    res_plot = post_plot - pre_plot

    # Calculate vmax and common plot parameters
    vmax = float(
        np.nanmax(np.array([np.nanmax(pre_plot), np.nanmax(post_plot)]))
    )
    res_v = float(np.nanmax(np.abs(res_plot)))
    
    # Build common parameters with defaults
    common = {}
    if pixel_scale is not None:
        common['pixel_scale'] = pixel_scale
    else:
        common['pixel_scale'] = dlu.rad2arcsec(deconvolution_model.pscale_in)
    
    if roll_angle_degrees is not None:
        common['roll_angle_degrees'] = roll_angle_degrees
    else:
        common['roll_angle_degrees'] = -ois[0].parang
    
    if diff_lim is not None:
        common['diff_lim'] = diff_lim
    else:
        common['diff_lim'] = dlu.rad2arcsec(ois[0].wavel / optics_diam)
    
    if scale is not None:
        common['scale'] = scale

    # Get history data
    log_dist_history = deconvolution_result.history["log_dist"][filter]
    contrast_hist = deconvolution_result.history["contrast"][filter]
    loss = deconvolution_result.losses[0]

    # Stretched deconvolution image
    stretched_pre = drt.plotting.plot_result(
        ax_dict["stretched_decon"],
        pre_plot,
        cmap=inferno,
        norm=mpl.colors.PowerNorm(stretch_amount, vmin=0, vmax=vmax),
        **common,
    )
    ax_dict["stretched_decon"].scatter(
        [0], [0], marker="*", color="white", s=10
    )
    ax_dict["stretched_decon"].set(
        title=f"{target} - {filter} - Deconvolution - Stretched"
    )
    fig.colorbar(
        stretched_pre,
        cax=ax_dict["sd_cbar"],
        orientation="horizontal",
        ticks=mpl.ticker.MaxNLocator(nbins=5),
    )

    # Deconvolution image
    pre = drt.plotting.plot_result(
        ax_dict["decon"],
        pre_plot,
        cmap=inferno,
        norm=mpl.colors.PowerNorm(1.0, vmin=0, vmax=vmax),
        **common,
    )
    ax_dict["decon"].scatter([0], [0], marker="*", color="white", s=10)
    ax_dict["decon"].set(title=f"{target} - {filter} - Deconvolution")
    fig.colorbar(
        pre,
        cax=ax_dict["d_cbar"],
        orientation="horizontal",
        ticks=mpl.ticker.MaxNLocator(nbins=5),
    )

    # Stretched BFGS image
    stretched_post = drt.plotting.plot_result(
        ax_dict["stretched_bfgs"],
        post_plot,
        cmap=inferno,
        norm=mpl.colors.PowerNorm(stretch_amount, vmin=0, vmax=vmax),
        **common,
    )
    ax_dict["stretched_bfgs"].scatter([0], [0], marker="*", color="white", s=10)
    ax_dict["stretched_bfgs"].set(
        title=f"{target} - {filter} - BFGS - Stretched"
    )
    fig.colorbar(
        stretched_post,
        cax=ax_dict["sbfgs_cbar"],
        orientation="horizontal",
        ticks=mpl.ticker.MaxNLocator(nbins=5),
    )

    # BFGS image
    post = drt.plotting.plot_result(
        ax_dict["bfgs"],
        post_plot,
        cmap=inferno,
        norm=mpl.colors.PowerNorm(1.0, vmin=0, vmax=vmax),
        **common,
    )
    ax_dict["bfgs"].scatter([0], [0], marker="*", color="white", s=10)
    ax_dict["bfgs"].set(title=f"{target} - {filter} - BFGS")
    fig.colorbar(
        post,
        cax=ax_dict["bfgs_cbar"],
        orientation="horizontal",
        ticks=mpl.ticker.MaxNLocator(nbins=5),
    )

    # Residual image
    c_res = drt.plotting.plot_result(
        ax_dict["residual"],
        res_plot,
        cmap=coolwarm,
        norm=mpl.colors.Normalize(vmin=-res_v, vmax=res_v),
        **common,
    )
    ax_dict["residual"].scatter([0], [0], marker="*", color="black", s=10)
    ax_dict["residual"].set(title=f"{target} - {filter} - Residual")
    fig.colorbar(
        c_res,
        cax=ax_dict["r_cbar"],
        orientation="horizontal",
        ticks=mpl.ticker.MaxNLocator(nbins=5),
    )

    # Contrast history
    ax_dict["contrast"].plot(contrast_hist)
    ax_dict["contrast"].set(
        title=f"{target} - {filter} - Contrast History",
        xlabel="Epoch",
        ylabel="Contrast",
    )

    # Latent history
    ax_dict["latent"].plot(log_dist_history, color="tab:blue", alpha=0.5)
    ax_dict["latent"].set(
        title=f"{target} - {filter} - Latent History",
        xlabel="Epoch",
        ylabel="Latents",
    )

    # Loss history
    ax_dict["loss"].plot(loss)
    ax_dict["loss"].set(
        title=f"{target} - Deconvolution Loss", xlabel="Epoch", ylabel="Loss"
    )

    start = int(final_loss_percent * len(loss))
    epochs_range = np.arange(start, len(loss))
    ax_dict["fin_loss"].plot(epochs_range[:-1], loss[start:-1])
    ax_dict["fin_loss"].set_title(
        f"{target} - {filter} - Final Deconvolution Loss"
    )
    ax_dict["fin_loss"].set_xlabel("Epoch")
    ax_dict["fin_loss"].set_ylabel("Loss")

    # Text box with parameters and results
    ax_dict["text"].axis("off")
    fancy_box = FancyBboxPatch(
        (0.02, 0.02),
        0.96,
        0.96,
        boxstyle="round,pad=0.02",
        transform=ax_dict["text"].transAxes,
        facecolor="none",
        edgecolor="black",
        linewidth=2,
    )
    ax_dict["text"].add_patch(fancy_box)
    ax_dict["text"].text(
        0.1,
        0.5,
        f"Target:              {target}\n"
        f"Filter:              {filter}\n"
        f"Epochs:              {epochs}\n"
        f"Latent LR:           {log_dist_lr}\n"
        f"Contrast LR:         {contrast_lr}\n"
        f"Prior:               {prior}\n"
        f"Loss Before BFGS:    {bfgs_result['loss_before']:.3e}\n"
        f"Loss After BFGS:     {bfgs_result['loss_after']:.3e}\n"
        f"Grad Inf Before:     {bfgs_result['grad_inf_before']:.3e}\n"
        f"Grad Inf After:      {bfgs_result['grad_inf_after']:.3e}",
        transform=ax_dict["text"].transAxes,
        fontsize=20,
    )

    fig.tight_layout()

    # Manually position colorbars below their respective images
    stretched_imgs = ["stretched_decon", "stretched_bfgs"]
    s_cbar_dict = {"stretched_decon": "sd_cbar", "stretched_bfgs": "sbfgs_cbar"}
    imgs = ["decon", "bfgs", "residual"]
    cbar_dict = {"decon": "d_cbar", "bfgs": "bfgs_cbar", "residual": "r_cbar"}

    for si in stretched_imgs:
        img_ax = ax_dict[si]
        cb_ax = ax_dict[s_cbar_dict[si]]
        img_pos = img_ax.get_position()
        cb_ax.set_position([
            img_pos.x0,
            img_pos.y0 - 0.04,
            img_pos.width,
            0.012,
        ])

    for i in imgs:
        img_ax = ax_dict[i]
        cb_ax = ax_dict[cbar_dict[i]]
        img_pos = img_ax.get_position()
        cb_ax.set_position([
            img_pos.x0,
            img_pos.y0 - 0.04,
            img_pos.width,
            0.012,
        ])

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax_dict