# %%

from utils import run_lstsq, get_sae_info, sae_info_to_params, layer_to_sae_ids
import pickle
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from tqdm import tqdm
from utils import BASE_DIR
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ['OMP_NUM_THREADS'] = '1'

# %%

import glob

# Get all files ending in active_sae_features_layer_5_res.pt in BASE_DIR
active_feature_files = glob.glob(os.path.join(BASE_DIR, "**/*active_sae_features_layer_*_res.pt"), recursive=True)
all_sae_ids = []
all_models = []
all_layers = []
for f in active_feature_files:
    all_models.append(f.split('_sae_scaling/')[0].split('/')[-1])
    all_sae_ids.append(f.split('sae_scaling/')[1].split('/active_sae_features_layer_')[0])
    all_layers.append(int(f.split('layer_')[-1].split('_')[0]))

print(len(all_sae_ids))
print([sae_id for sae_id in all_sae_ids if "llama" in sae_id])
# %%
device = 'cuda:1'
use_acts = True

shuffle_order = True
order = np.random.permutation(len(all_sae_ids))
reordered_all_sae_ids = [all_sae_ids[i] for i in order]
reordered_all_models = [all_models[i] for i in order]
reordered_all_layers = [all_layers[i] for i in order]

for sae_id, model, layer in zip(reordered_all_sae_ids, reordered_all_models, reordered_all_layers):
    file_name = f"{BASE_DIR}/data/percents/{sae_id}/{model}_{layer}_{'self' if not use_acts else 'acts'}.pkl"
    os.makedirs(f"{BASE_DIR}/data/percents/{sae_id}", exist_ok=True)
    if os.path.exists(file_name):
        continue

    print(f"Processing {model} {sae_id}")

    sae_info = get_sae_info(sae_name=sae_id, model=model, layer=layer)
    x = sae_info.acts_flattened
    y = -sae_info.sae_error_vecs_flattened
    unexplained_noise = y - x @ run_lstsq(x, y, device=device, lstsq_token_threshold="all")[2]

    percents = {} 
    for x, name in [
        (sae_info.acts_flattened, "x"), 
        (sae_info.reconstruction_vecs_flattened, "SAE(x)"), 
        (-sae_info.sae_error_vecs_flattened, "SAE Error Vector"), 
        (unexplained_noise, "Nonlinear Error"), 
        (-sae_info.sae_error_vecs_flattened-unexplained_noise, "Linear Error")
    ]:
        include_ones = True
        y = x.norm(dim=-1)**2
        if include_ones:
            x = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        if use_acts:
            norm_prediction_percent = run_lstsq(sae_info.acts_flattened, x.norm(dim=-1)**2, device=device, randomize_order=True)[1]
        else:
            norm_prediction_percent = run_lstsq(x, x.norm(dim=-1)**2, device=device, randomize_order=True)[1]
        percents[name] = norm_prediction_percent
    
    with open(file_name, "wb") as f:
        pickle.dump(percents, f)

# %%
use_acts = True
metadata_to_percents = {}
for sae_id, model, layer in zip(all_sae_ids, all_models, all_layers):
    file_name = f"{BASE_DIR}/data/percents/{sae_id}/{model}_{layer}_{'self' if not use_acts else 'acts'}.pkl"
    if not os.path.exists(file_name):
        continue
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    metadata_to_percents[(sae_id, model, layer)] = data

print(len(metadata_to_percents))

# %%

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 6,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
})

use_acts = True
# Create the plot
fig, ax = plt.subplots(figsize=(2.75, 2.5))

legend_elements = []


norm_prediction_names = ["x", "SAE(x)", "SAE Error Vector", "Linear Error", "Nonlinear Error"]
norm_prediction_latex_names = ["$x$", "$Sae(x)$", "$SaeError(x)$", "$LinearError(x)$", "$NonlinearError(x)$"]
norm_prediction_vals = [[] for _ in range(len(norm_prediction_names))]

# Collect values from metadata_to_percents
for metadata, percents in metadata_to_percents.items():
    for name in norm_prediction_names:
        if name in percents:
            percent = percents[name]
            if percent > 0.5:
                norm_prediction_vals[norm_prediction_names.index(name)].append(percent)

# Create violin plots
parts = ax.violinplot(norm_prediction_vals, showmeans=False, showmedians=False, showextrema=False)

# Customize violin plots
for i, pc in enumerate(parts['bodies']):
    color = plt.cm.viridis(i / len(norm_prediction_vals))
    pc.set_facecolor(color)
    pc.set_edgecolor('black')
    pc.set_alpha(0.7 if use_acts else 0.2)
    
    if not use_acts:
        pc.set_hatch('///')

# Add legend element
legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=plt.cm.viridis(0.5), 
                                   alpha=0.7 if use_acts else 0.2,
                                   hatch='///' if not use_acts else None,
                                   label='Predict from x' if use_acts else 'Predict from self'))

if use_acts:
    plt.ylim(0.55, 1.03)
else:
    plt.ylim(-0.1, 1.03)

# Customize the plot
ax.set_ylabel('Norm Prediction Test $R^2$')
ax.set_xticks(np.arange(1, len(norm_prediction_names) + 1))
ax.tick_params(axis='both', which='major')
ax.set_xticklabels(norm_prediction_latex_names, rotation=45, ha='right')

# Add legend
# ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=8)

# Save the figure
plt.tight_layout()
plt.savefig("plots/norm_prediction_test.pdf", bbox_inches='tight')

plt.show()
plt.close()
# %%



plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 5,
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 5.5,
    'ytick.labelsize': 5.5,
    'legend.fontsize': 6,
})


def get_sae_ids_closest_to_target_l0(model_name, layer_type, target_l0, layers, width):
    all_layer_sae_ids = []
    for layer in layers:
        sae_ids = layer_to_sae_ids(layer, model_name, layer_type)
        correct_width_sae_ids = []
        for sae_id in sae_ids:
            params = sae_info_to_params(sae_id, model_name)
            if params["width_str"] == width:
                correct_width_sae_ids.append(sae_id)
        closest_l0_sae_id = min(correct_width_sae_ids, key=lambda x: abs(sae_info_to_params(x, model_name)["l0"] - target_l0))
        all_layer_sae_ids.append(closest_l0_sae_id)
    return all_layer_sae_ids

plt.figure(figsize=(2.75, 2))

# colors = plt.cm.plasma(np.linspace(0, 1, 8))
colors = plt.cm.tab10(np.linspace(0, 1, 10))
colors_to_use = [
    colors[0], 
    colors[1], 
    colors[2], 
    colors[3], 
    colors[4], 
    colors[9]]
color_id = 0
for model_name, widths, layers in [("llama_3.1_8b", ["8x", "32x"], range(32)), ("gemma_2_2b", ["16k", "65k"], range(26)), ("gemma_2_9b", ["16k", "131k"], range(41))]:
    for width in widths:
        target_l0 = 50
        all_layer_sae_ids = get_sae_ids_closest_to_target_l0(model_name, "res", target_l0, layers, width)
        
        x_values = []  # Percent through model
        r2_values = [] # R^2 values
        
        for i, sae_id in enumerate(all_layer_sae_ids):
            params = sae_info_to_params(sae_id, model_name)
            layer = params["layer"]
            print(sae_id, model_name, layer)
            if (sae_id, model_name, layer) in metadata_to_percents:
                r2_value = metadata_to_percents[(sae_id, model_name, layer)]["SAE Error Vector"]
                x_values.append(layer / len(layers))
                r2_values.append(max(0, r2_value))

        
        if len(x_values) > 0:
            if width == "8x":
                width = "32k"
            elif width == "32x":
                width = "131k"
            plt.plot(x_values, r2_values, label=f"{model_name} {width}", marker='o', alpha=0.6, color=colors_to_use[color_id], markersize=4)
            color_id += 1
plt.xlabel("Layer by depth fraction in model")
plt.ylabel("$R^2$ predicting error norm")
# plt.title("R² Score vs Layer Depth")
plt.legend()
plt.ylim(0.4, 1.01)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/r2_vs_depth.pdf", bbox_inches='tight')
plt.show()
plt.close()

# %%
