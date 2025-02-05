# %%

import transformer_lens
from sae_lens import SAE
from utils import load_sae
from datasets import load_dataset
import torch
from tqdm import tqdm
import os
import argparse
import numpy as np
from utils import BASE_DIR

torch.set_grad_enabled(False)


# %%

argparser = argparse.ArgumentParser()
argparser.add_argument("--device", type=str, default="cuda:0")
argparser.add_argument("--model_name", type=str, default="gemma_2_2b", choices=["gemma_2_2b", "gemma_2_9b", "llama_3.1_8b"])
argparser.add_argument("--sae_id", type=str, required=True)
argparser.add_argument("--layer", type=int, required=True)
argparser.add_argument("--layer_type", choices=["res", "mlp", "att"], default="res")

args = argparser.parse_args()

model_name = args.model_name
sae_id = args.sae_id
device = args.device
layer = args.layer
layer_type = args.layer_type


if layer_type != "res":
    save_dir_base = f"{BASE_DIR}/{model_name}_sae_scaling_{args.layer_type}"
else:
    save_dir_base = f"{BASE_DIR}/{model_name}_sae_scaling"
save_dir = f"{save_dir_base}/{sae_id}"
os.makedirs(save_dir, exist_ok=True)

# %%


all_files_exist = all([
    os.path.exists(f"{save_dir}/sae_l0s_layer_{layer}_{layer_type}.pt"),
    os.path.exists(f"{save_dir}/sae_errors_layer_{layer}_{layer_type}.pt"),
    os.path.exists(f"{save_dir}/sae_error_vecs_layer_{layer}_{layer_type}.pt"),
    os.path.exists(f"{save_dir}/feature_act_norms_layer_{layer}_{layer_type}.pt"),
    os.path.exists(f"{save_dir}/active_sae_features_layer_{layer}_{layer_type}.pt"),
])

if all_files_exist:
    print(f"All files exist for {sae_id}, skipping")
    exit()


# %%

if layer_type != "att":
    if model_name == "gemma_2_2b":
        acts_shape = [300, 1024, 2304]
    elif model_name == "gemma_2_9b":
        acts_shape = [300, 1024, 3584]
    elif model_name == "llama_3.1_8b":
        acts_shape = [300, 1024, 4096]
else:
    if model_name != "gemma_2_9b":
        raise NotImplementedError()
    else:
        acts_shape = [300, 1024, 16, 256]
acts = torch.load(f"{save_dir_base}/acts_layer_{layer}_{layer_type}.pt").view(*acts_shape)

# %%
sae = load_sae(sae_id, model_name=model_name, layer_type=layer_type).to(device)
hook_name = sae.cfg.hook_name
print(hook_name)

# %%

batch_size = 1
if layer_type != "att":
    num_contexts, ctx_len, dim = acts.shape
else:
    num_contexts, ctx_len, num_heads, head_dim = acts.shape
    
total_tokens = num_contexts * ctx_len
print(f"Total tokens: {total_tokens / 1e6:.2f}M")

# %%


all_sae_l0s = []
all_sae_errors = []
all_feature_act_norms = []
all_sae_error_vecs = []
all_sae_features_acts = []


def save_so_far():
    all_sae_l0s_cat = torch.cat(all_sae_l0s, dim=0)
    torch.save(all_sae_l0s_cat, f"{save_dir}/sae_l0s_layer_{layer}_{layer_type}.pt")
    all_sae_errors_cat = torch.cat(all_sae_errors, dim=0)
    torch.save(all_sae_errors_cat, f"{save_dir}/sae_errors_layer_{layer}_{layer_type}.pt")
    all_sae_error_vecs_cat = torch.cat(all_sae_error_vecs, dim=0)
    torch.save(all_sae_error_vecs_cat, f"{save_dir}/sae_error_vecs_layer_{layer}_{layer_type}.pt")
    all_feature_act_norms_cat = torch.cat(all_feature_act_norms, dim=0)
    torch.save(
        all_feature_act_norms_cat, f"{save_dir}/feature_act_norms_layer_{layer}_{layer_type}.pt"
    )

    torch.save(all_sae_features_acts, f"{save_dir}/active_sae_features_layer_{layer}_{layer_type}.pt")


# %%

bar = tqdm(range(0, num_contexts, batch_size))
for i in bar:
    acts_batch_cpu = acts[i : i + batch_size]
    acts_batch = acts_batch_cpu.clone().to(device)

    feature_acts = sae.encode(acts_batch)
    reconstructions = sae.decode(feature_acts)
    feature_acts_cpu = feature_acts.to("cpu")
    l0s = (feature_acts_cpu > 0).sum(dim=-1)
    feature_acts_norms = feature_acts_cpu.norm(dim=-1)

    for j in range(batch_size):
        feature_acts_j = feature_acts_cpu[j]
        nonzero_feature_indices_j = torch.nonzero(feature_acts_j, as_tuple=True)
        nonzero_feature_values_j = feature_acts_j[nonzero_feature_indices_j]
        res = [nonzero_feature_indices_j[0], nonzero_feature_indices_j[1], nonzero_feature_values_j]
        all_sae_features_acts.append(res)

    # Get reconstruction error
    reconstructions_cpu = reconstructions.to("cpu")
    reconstruction_errors = reconstructions_cpu - acts_batch_cpu
    reconstruction_error_norms = reconstruction_errors.square().sum(dim=-1)

    all_sae_l0s.append(l0s.to("cpu"))
    all_sae_errors.append(reconstruction_error_norms.to("cpu"))
    all_sae_error_vecs.append(reconstruction_errors.to("cpu"))
    all_feature_act_norms.append(feature_acts_norms.to("cpu"))

    bar.set_description(
        f"Num tokens: {len(all_sae_errors) * batch_size * ctx_len / 1e6:.2f}M"
    )

save_so_far()

# %%
