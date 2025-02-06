# %%

import matplotlib
from matplotlib.patches import ConnectionPatch
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import run_lstsq, fraction_variance_unexplained, BASE_DIR, get_sae_info, layer_to_sae_ids, sae_info_to_params
import pickle
from scipy.interpolate import griddata
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from matplotlib.cm import get_cmap
from utils import normalized_mse
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import argparse
from tqdm import tqdm
torch.set_grad_enabled(False)

import os
os.makedirs("plots", exist_ok=True)
os.environ["OMP_NUM_THREADS"] = "10"

# %%


def make_plots(model, layer, method="linear"):

    with open(f"{BASE_DIR}/data/sae_power_laws_{model}_{layer}.pkl", "rb") as f:
        sae_ids, normalized_mses, normalized_mses_with_predictions, fvus, fvus_with_predictions, empirical_l0s, sae_error_norm_r_squareds, sae_error_vec_r_squareds = pickle.load(f)

    plot_type = "fvu"
    # plot_type = "mse"

    if plot_type == "mse":
        losses = normalized_mses
        losses_with_preds = normalized_mses_with_predictions
        ylabel = "Normalized MSE"
    else:
        losses = fvus
        losses_with_preds = fvus_with_predictions
        ylabel = "FVU"

    if "gemma" in model:
        sae_widths = ["16k", "32k", "65k", "131k", "262k", "524k", "1m"]
    else:
        sae_widths = ["8x", "32x"]

    # Sort saes
    correct_order_saes = []
    correct_order_losses = []
    correct_order_losses_with_preds = []
    correct_order_empirical_l0s = []
    correct_order_sae_error_norm_r_squareds = []
    correct_order_sae_error_vec_r_squareds = []
    for target_width in sae_widths:
        for i, sae_id in enumerate(sae_ids):
            width = sae_info_to_params(sae_id, model)["width_str"]
            if width == target_width:
                correct_order_saes.append(sae_id)
                correct_order_losses.append(losses[i])
                correct_order_losses_with_preds.append(losses_with_preds[i])
                correct_order_empirical_l0s.append(empirical_l0s[i])
                correct_order_sae_error_norm_r_squareds.append(sae_error_norm_r_squareds[i])
                correct_order_sae_error_vec_r_squareds.append(sae_error_vec_r_squareds[i])

    saes = correct_order_saes
    losses = correct_order_losses
    losses_with_preds = correct_order_losses_with_preds
    empirical_l0s = correct_order_empirical_l0s
    sae_error_norm_r_squareds = correct_order_sae_error_norm_r_squareds
    sae_error_vec_r_squareds = correct_order_sae_error_vec_r_squareds

    if "gemma" in model:
        widths = [2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20]
    else:
        widths = [4096 * 8, 4096 * 32]

    # Create figure with 3 subplots horizontally
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.5), sharey=True)

    # Helper function to create contour data
    def create_contour_data(widths, saes, z_data):
        all_widths = []
        all_z = []
        all_l0s = []
        for sae, z_local in zip(saes, z_data):
            params = sae_info_to_params(sae, model)
            all_widths.append(params["width"])
            all_z.append(z_local)
            all_l0s.append(params["l0"])
        return np.array(all_widths), np.array(all_l0s), np.array(all_z)

    # First plot - FVU
    all_widths = []
    all_losses_with_preds = []
    all_l0s = []
    for sae, losses_with_preds_local in zip(saes, losses_with_preds):
        params = sae_info_to_params(sae, model)
        all_widths.append(params["width"])
        all_losses_with_preds.append(losses_with_preds_local)
        all_l0s.append(params["l0"])

    x_contour = np.array(all_widths)
    y_contour = np.array(all_l0s)
    z_contour = np.array(all_losses_with_preds)[:, 0]

    x_min, x_max = x_contour.min() / 1.15, x_contour.max() * 1.15
    y_min, y_max = y_contour.min() / 1.15, y_contour.max() * 1.15
    
    xi = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    yi = np.logspace(np.log10(y_min), np.log10(y_max), 100)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((x_contour, y_contour), z_contour, (XI, YI), method=method)

    contour = axes[2].contourf(XI, YI, ZI, levels=15, cmap='viridis')
    cbar = plt.colorbar(contour, ax=axes[2])
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(label=f"Nonlinear SAE {ylabel}", size=7, y=0.44)
    axes[2].scatter(x_contour, y_contour, c='black', s=1, alpha=0.7)
    axes[2].set_xlabel("SAE Width", fontsize=8)
    # axes[2].set_ylabel('SAE L0', fontsize=8)
    axes[2].tick_params(axis='both', which='major', labelsize=7)
    axes[2].tick_params(axis='both', which='minor', labelsize=7)
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')

    # Second and third plots - R squared
    all_zs = [sae_error_norm_r_squareds, sae_error_vec_r_squareds]
    labels = ["norm", "vec"]

    for i, (z_data, label) in enumerate(zip(all_zs, labels)):
        # i -= 1
        x_contour, y_contour, z_contour = create_contour_data(widths, saes, z_data)
        z_contour = z_contour[:, 0]
        
        ZI = griddata((x_contour, y_contour), z_contour, (XI, YI), method=method)
        
        contour = axes[i].contourf(XI, YI, ZI, levels=15, cmap='viridis', extend='both')
        cbar = plt.colorbar(contour, ax=axes[i])
        cbar.ax.tick_params(labelsize=6)
        
        axes[i].scatter(x_contour, y_contour, c='black', s=1, alpha=0.7)
        axes[i].set_xlabel("SAE Width", fontsize=8)
        if i == 0:  # Only show y label on leftmost plot
            axes[i].set_ylabel('SAE L0', fontsize=8)
        axes[i].tick_params(axis='both', which='major', labelsize=7)
        axes[i].tick_params(axis='both', which='minor', labelsize=7)
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')
        axes[i].set_xlim(x_min, x_max)
        axes[i].set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f"plots/sae_power_law_contours_{model}_{layer}_combined.pdf", bbox_inches='tight', pad_inches=0.02)
    plt.show()
    plt.close()

model_to_layers = {
    "gemma_2_9b": [9, 20, 31],
    "gemma_2_2b": [5, 12, 19]
}
for model in model_to_layers:
    for layer in model_to_layers[model]:
        try:
            make_plots(model, layer)
        except Exception as e:
            print(f"Error making plots for {model} {layer}: {e}")

# %%