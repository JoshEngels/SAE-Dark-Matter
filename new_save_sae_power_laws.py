# %%
# python3 sae_power_laws.py --model gemma_2_2b --layer 5 ; python3 sae_power_laws.py --model gemma_2_2b --layer 19 ; python3 sae_power_laws.py --model gemma_2_2b --layer 12 ; 
# python3 sae_power_laws.py --model gemma_2_9b --layer 9 ; python3 sae_power_laws.py --model gemma_2_9b --layer 20

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

device = "cuda:0"
to_plot = "both"
layer = 20
model = "gemma_2_9b"

# %%

argparser = argparse.ArgumentParser()
argparser.add_argument("--device", type=str, default="cuda:0")
argparser.add_argument("--to_plot", choices=["both", "pursuit"], default="both")
argparser.add_argument("--model", type=str, default="gemma_2_2b")
argparser.add_argument("--layer", type=int, default=12)
args = argparser.parse_args()

device = args.device
to_plot = args.to_plot
layer = args.layer
model = args.model

# %%

os.makedirs(f"{BASE_DIR}/data", exist_ok=True)
save_file = f"{BASE_DIR}/data/sae_power_laws_{model}_{layer}.pkl"

if not os.path.exists(save_file):

    sae_ids = layer_to_sae_ids(layer, model)

    normalized_mses = [[]]
    normalized_mses_with_predictions = [[]]
    fvus = [[]]
    fvus_with_predictions = [[]]
    sae_error_norm_r_squareds = [[]]
    sae_error_vec_r_squareds = [[]]
    empirical_l0s = [[]]

    for sae_id in tqdm(sae_ids):
        sae_info = get_sae_info(layer=layer, sae_name=sae_id, model=model)

        normalized_mses[-1].append(normalized_mse(sae_info.reconstruction_vecs_flattened, sae_info.acts_flattened))
        fvus[-1].append(fraction_variance_unexplained(sae_info.acts_flattened, sae_info.reconstruction_vecs_flattened))

        empirical_l0 = 0
        for active_feature_instance in sae_info.active_sae_features:
            num_zero_indices = (active_feature_instance[0] == 0).sum()
            empirical_l0 += len(active_feature_instance[0]) - num_zero_indices.item()
        empirical_l0 /= len(sae_info.active_sae_features) * 823
        empirical_l0s[-1].append(empirical_l0)

        x_gpu = sae_info.acts_flattened.to(device)
        y_gpu = -sae_info.sae_error_vecs_flattened.to(device)
        _, r_squared, sol = run_lstsq(x_gpu, y_gpu, device=device)
        sae_error_vec_r_squareds[-1].append(r_squared)

        predictions = x_gpu @ sol.to(device)
        normalized_mses_with_predictions[-1].append(normalized_mse(sae_info.reconstruction_vecs_flattened + predictions.to("cpu"), sae_info.acts_flattened))
        fvus_with_predictions[-1].append(fraction_variance_unexplained(sae_info.acts_flattened, sae_info.reconstruction_vecs_flattened + predictions.to("cpu")))

        y_gpu = y_gpu.norm(dim=-1) ** 2
        _, r_squared, _ = run_lstsq(x_gpu, y_gpu, device=device)
        sae_error_norm_r_squareds[-1].append(r_squared)

        normalized_mses.append([])
        fvus.append([])
        normalized_mses_with_predictions.append([])
        fvus_with_predictions.append([])
        empirical_l0s.append([])
        sae_error_norm_r_squareds.append([])
        sae_error_vec_r_squareds.append([])

    to_save = (sae_ids, normalized_mses[:-1], normalized_mses_with_predictions[:-1], fvus[:-1], fvus_with_predictions[:-1], empirical_l0s[:-1], sae_error_norm_r_squareds[:-1], sae_error_vec_r_squareds[:-1])

    with open(save_file, "wb") as f:
        pickle.dump(to_save, f)

