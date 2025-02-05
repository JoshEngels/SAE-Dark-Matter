# %%

import torch
import numpy as np
import os
import dill as pickle
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens.sae import SAE


try:
    from IPython import get_ipython  # type: ignore

    ipython = get_ipython()
    assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

    is_notebook = True
except:
    is_notebook = False

base_dirs_to_try = ["/mnt/sdb/jengels"]
BASE_DIR = None
for dir in base_dirs_to_try:
    if os.path.exists(dir):
        BASE_DIR = dir
        break
if BASE_DIR is None:
    raise ValueError("No base directory found, please add a directory to the list of base_dirs_to_try in utils.py")

def calculate_r_squared(target_variable, residuals):
    total_sum_of_squares = torch.sum(
        (target_variable - torch.mean(target_variable, dim=0)) ** 2, dim=0
    )
    residual_sum_of_squares = torch.sum(residuals**2, dim=0)
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    # If there are multiple targets, take the mean R^2
    # This is a bit of a hack, should probably find the variance explained in a more principled way
    return r_squared.mean().item()


def run_lstsq(
    explainer_variables,
    target_variable,
    verbose=False,
    lstsq_token_threshold=150_000,
    eval_token_threshold=300_000,
    randomize_order=True,
    device="cuda:1",
    rcond=1e-9
):
    
    assert randomize_order, "Randomize order should be True otherwise R^2 might be biased, it's still an option so if you reaaaaly need it you can comment out this line"

    if randomize_order:
        random_indices = torch.randperm(explainer_variables.shape[0])
        explainer_variables = explainer_variables[random_indices]
        target_variable = target_variable[random_indices]

    if lstsq_token_threshold == "all":
        lstsq_token_threshold = explainer_variables.shape[0]

    target_for_regression = target_variable[:lstsq_token_threshold].to(device)
    explainer_variables_for_regression = explainer_variables[:lstsq_token_threshold].to(
        device
    )

    solution = torch.linalg.lstsq(
        explainer_variables_for_regression, target_for_regression, rcond=rcond
    ).solution

    if verbose:
        print(solution)

    if verbose:
        train_residuals = (
            target_for_regression - explainer_variables_for_regression @ solution
        )
        print(train_residuals)
        print("Train R^2", calculate_r_squared(target_for_regression, train_residuals))

    target_for_eval = target_variable[lstsq_token_threshold:eval_token_threshold].to(
        device
    )
    if len(target_for_eval.shape) > 0:
        explainer_variables_for_eval = explainer_variables[
            lstsq_token_threshold:eval_token_threshold
        ].to(device)

        # Calculate residuals
        residuals = target_for_eval - explainer_variables_for_eval @ solution

        # Calculate R^2
        r_squared = calculate_r_squared(target_for_eval, residuals)
        if verbose:
            print(f"R^2: {r_squared}")
    else:
        r_squared = None

    return residuals.cpu(), r_squared, solution.cpu()


class SAEInfoObject:
    def __init__(
        self,
        model_loss_flattened,
        sae_error_norms_flattened,
        l0s_flattened,
        act_norm_flattened,
        tokens_flattened,
        ones,
        max_token,
        acts_flattened,
        sae_error_vecs_flattened,
        active_sae_features,
        context_start_threshold,
        context_end_threshold,
        sae_name,
        model_name
    ):
        self.model_loss_flattened = model_loss_flattened
        self.sae_error_norms_flattened = sae_error_norms_flattened
        self.l0s_flattened = l0s_flattened
        self.act_norm_flattened = act_norm_flattened
        self.tokens_flattened = tokens_flattened
        self.ones = ones
        self.max_token = max_token
        self.acts_flattened = acts_flattened
        self.sae_error_vecs_flattened = sae_error_vecs_flattened
        self.active_sae_features = active_sae_features
        self.context_start_threshold = context_start_threshold
        self.context_end_threshold = context_end_threshold
        self.sae_name = sae_name

        if "gemma" in model_name:
            layer, width, average_l0 = sae_name.split("/")
            layer = int(layer.split("_")[-1])
            average_l0 = int(average_l0.split("_")[-1])
            width = {"16k": int(2**14), "32k": int(2**15), "65k": int(2**16), "131k": int(2**17), "262k": int(2**18), "524k": int(2**19), "1m": int(2**20)}[width.split("_")[-1]]
            self.layer = layer
            self.width = width
            self.average_l0 = average_l0

        self.reconstruction_vecs_flattened = sae_error_vecs_flattened + acts_flattened

    def get_feature_freqs(self):

        feature_frequencies = torch.zeros(self.width, dtype=torch.float32)
        average_feature_values = torch.zeros(self.width, dtype=torch.float32)
        for tokens_indices, feature_indices, feature_values in self.active_sae_features:
            token_index_within_range = (tokens_indices >= self.context_start_threshold) & (tokens_indices < self.context_end_threshold)

            valid_feature_indices = feature_indices[token_index_within_range]
            ones = torch.ones_like(valid_feature_indices, dtype=torch.float32)
            feature_frequencies.index_add_(0, valid_feature_indices, ones)

            to_add = feature_values[torch.nonzero(token_index_within_range).flatten()]
            average_feature_values.index_add_(0, valid_feature_indices, to_add)


        average_feature_values_when_active = average_feature_values.clone()
        average_feature_values_when_active /= feature_frequencies
        average_feature_values_overall = average_feature_values.clone()
        average_feature_values_overall /= (self.context_end_threshold - self.context_start_threshold) * len(self.active_sae_features)

        return feature_frequencies, average_feature_values_when_active, average_feature_values_overall



def get_gemma_2_9b_sae_ids(layer, layer_type="res"):
    assert layer_type in ["res", "att", "mlp"]
    all_gemma_scope_saes = get_pretrained_saes_directory()[f"gemma-scope-9b-pt-{layer_type}"].saes_map
    all_sae_ids = [sae_id for sae_id in all_gemma_scope_saes if sae_id.split("/")[0] == f"layer_{layer}"]
    return all_sae_ids

def get_gemma_2_2b_sae_ids(layer, layer_type="res"):
    assert layer_type in ["res", "att", "mlp"]
    all_gemma_scope_saes = get_pretrained_saes_directory()[f"gemma-scope-2b-pt-{layer_type}"].saes_map
    all_sae_ids = [sae_id for sae_id in all_gemma_scope_saes if sae_id.split("/")[0] == f"layer_{layer}"]
    return all_sae_ids

def get_llama_3_1_8b_sae_ids(layer, layer_type="res"):
    assert layer_type == "res"
    return [f"l{layer}r_8x", f"l{layer}r_32x"]

def layer_to_sae_ids(layer, model_name, layer_type="res"):
    if model_name == "gemma_2_2b":
        return get_gemma_2_2b_sae_ids(layer, layer_type)
    elif model_name == "gemma_2_9b":
        return get_gemma_2_9b_sae_ids(layer, layer_type)
    elif model_name == "llama_3.1_8b":
        return get_llama_3_1_8b_sae_ids(layer, layer_type)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

def load_sae(sae_id, model_name, layer_type="res"):
    if model_name == "gemma_2_2b":
        sae = SAE.from_pretrained(
            release = f"gemma-scope-2b-pt-{layer_type}",
            sae_id = sae_id,
            device = "cpu",
        )[0]
    elif model_name == "gemma_2_9b":
        sae = SAE.from_pretrained(
            release = f"gemma-scope-9b-pt-{layer_type}",
            sae_id = sae_id,
            device = "cpu",
        )[0]
    elif model_name == "llama_3.1_8b":
        assert layer_type == "res"
        width = sae_id.split("_")[-1]
        sae = SAE.from_pretrained(
            release = f"llama_scope_lxr_32x_{width}",
            sae_id = sae_id,
            device = "cpu",
        )[0]
    return sae

def gemma_sae_info_to_params(sae_id):
    layer, width, average_l0 = sae_id.split("/")
    layer = int(layer.split("_")[-1])
    average_l0 = int(average_l0.split("_")[-1])
    width_str = width.split("_")[-1]
    width = {"16k": int(2**14), "32k": int(2**15), "65k": int(2**16), "131k": int(2**17), "262k": int(2**18), "524k": int(2**19), "1m": int(2**20)}[width_str]
    return {
        "layer": layer,
        "width": width,
        "width_str": width_str,
        "l0": average_l0
    }

def llama_sae_info_to_params(sae_id):
    layer, width_str = sae_id.split("_")
    layer = int(layer.split("l")[-1][:-1])
    width = 4096 * int(width_str.split("x")[0])
    return {
        "layer": layer,
        "width": width,
        "width_str": width_str,
        "l0": 50
    }
def sae_info_to_params(sae_id, model_name):
    if model_name == "gemma_2_2b":
        return gemma_sae_info_to_params(sae_id)
    elif model_name == "gemma_2_9b":
        return gemma_sae_info_to_params(sae_id)
    elif model_name == "llama_3.1_8b":
        return llama_sae_info_to_params(sae_id)

# %%

def get_sae_info(layer, sae_name, model="gemma_2_2b", num_cols_start=None, layer_type="res") -> SAEInfoObject:
    layer_type_extension_load_dir = f"_{layer_type}" if layer_type != "res" else ""
    if "gemma_2" in model:
        model_size = model.split("_")[-1]
        base_load_dir = f"{BASE_DIR}/gemma_{model_size}_sae_scaling{layer_type_extension_load_dir}"
        if model_size == "2b":
            model_dim = 2304
        else:
            model_dim = 3584
        num_cols_end = 1023
        if num_cols_start is None:
            num_cols_start = 200
    elif model == "llama":
        base_load_dir = f"{BASE_DIR}/llama_8b_sae_scaling{layer_type_extension_load_dir}"
        model_dim = 4096
        num_cols_end = 2047
        if num_cols_start is None:
            num_cols_start = 200
    elif model == "gpt2":
        base_load_dir = f"{BASE_DIR}/gpt2_sae_scaling{layer_type_extension_load_dir}"
        model_dim = 768
        if num_cols_start is None:
            num_cols_start = 10
        num_cols_end = 511

    load_dir = f"{base_load_dir}/{sae_name}"

    l0s = torch.load(os.path.join(load_dir, f"sae_l0s_layer_{layer}.pt"))

    if layer_type != "att":
        sae_error_vecs = torch.from_file(
            f"{load_dir}/sae_error_vecs_layer_{layer}.npy",
            shared=False,
            size=l0s.numel() * model_dim,
        )
        sae_error_vecs = sae_error_vecs.view(l0s.shape[0], l0s.shape[1], model_dim)

        acts = torch.from_file(
            f"{base_load_dir}/acts_layer_{layer}.npy",
            shared=False,
            size=l0s.numel() * model_dim,
        )
        acts = acts.view(l0s.shape[0], l0s.shape[1], model_dim)
    else:
        assert model == "gemma_2_9b"
        num_heads = 16
        head_dim = 256
        sae_error_vecs = torch.from_file(
            f"{load_dir}/sae_error_vecs_layer_{layer}.npy",
            shared=False,
            size=l0s.numel() * num_heads * head_dim,
        )
        sae_error_vecs = sae_error_vecs.view(l0s.shape[0], l0s.shape[1], num_heads, head_dim)

        acts = torch.from_file(
            f"{base_load_dir}/acts_layer_{layer}.npy",
            shared=False,
            size=l0s.numel() * num_heads * head_dim,
        )
        acts = acts.view(l0s.shape[0], l0s.shape[1], num_heads, head_dim)

    sae_error_norms = torch.load(os.path.join(load_dir, f"sae_errors_layer_{layer}.pt"))
    feature_act_norms = torch.load(
        os.path.join(load_dir, f"feature_act_norms_layer_{layer}.pt")
    )

    model_losses = torch.load(os.path.join(base_load_dir, f"model_losses.pt"))
    tokens = torch.load(os.path.join(base_load_dir, f"tokens.pt"))
    act_norms = torch.load(os.path.join(base_load_dir, f"act_norms_layer_{layer}.pt"))

    l0s = l0s[:, num_cols_start:num_cols_end]
    sae_error_norms = sae_error_norms[:, num_cols_start:num_cols_end]
    sae_error_vecs = sae_error_vecs[:, num_cols_start:num_cols_end]
    feature_act_norms = feature_act_norms[:, num_cols_start:num_cols_end]
    model_losses = model_losses[:, num_cols_start:num_cols_end]
    tokens = tokens[:, num_cols_start:num_cols_end]
    act_norms = act_norms[:, num_cols_start:num_cols_end]

    print(acts.shape)
    acts = acts[:, num_cols_start:num_cols_end, :]

    model_loss_flattened = model_losses.flatten()
    sae_error_norms_flattened = sae_error_norms.flatten()
    l0s_flattened = l0s.flatten()
    act_norm_flattened = act_norms.flatten()
    tokens_flattened = tokens.flatten()

    # Flatten first and second dim
    acts_flattened = acts.flatten(0, 1)
    print(acts_flattened.shape)
    sae_error_vecs_flattened = sae_error_vecs.flatten(0, 1)

    ones = torch.ones_like(l0s_flattened)
    max_token = torch.max(tokens_flattened).item()

    active_sae_features = torch.load(
        os.path.join(load_dir, f"active_sae_features_layer_{layer}.pt")
    )

    return SAEInfoObject(
        model_loss_flattened,
        sae_error_norms_flattened,
        l0s_flattened,
        act_norm_flattened,
        tokens_flattened,
        ones,
        max_token,
        acts_flattened,
        sae_error_vecs_flattened,
        active_sae_features,
        num_cols_start,
        num_cols_end,
        sae_name,
        model
    )

def get_all_tokens(model="gemma_2_2b"):
    if "gemma_2" in model:
        model_size = model.split("_")[-1]
        base_load_dir = f"{BASE_DIR}/gemma_{model_size}_sae_scaling"
    elif model == "llama":
        base_load_dir = f"{BASE_DIR}/llama_8b_sae_scaling"
    return torch.load(f"{base_load_dir}/tokens.pt")


def get_sae_info_by_params(layer, sae_width, sae_l0, model="gemma_2_2b", num_cols_start=200, layer_type="res"):
    sae_name = f"layer_{layer}/width_{sae_width}/average_l0_{sae_l0}"
    return get_sae_info(layer, sae_name, model, num_cols_start, layer_type)

def normalized_mse(reconstruction, original_input):
    return (
        ((reconstruction - original_input) ** 2).mean(dim=1) / (original_input**2).mean(dim=1)
    ).mean()



def create_equal_length_bucket_encoding(x, num_buckets, buckets_to_use=None):
    # Ensure x is a 1D tensor
    x = x.squeeze()
    
    # Calculate min and max values
    min_val, max_val = x.min(), x.max()
    
    # Calculate bucket assignments
    bucket_assignments = torch.floor((x - min_val) / (max_val - min_val) * num_buckets).long()
    
    # Clip the assignments to ensure they're within the valid range
    bucket_assignments = torch.clamp(bucket_assignments, 0, num_buckets - 1)

    # Create one-hot encoding
    one_hot = torch.zeros(len(x), num_buckets)
    one_hot.scatter_(1, bucket_assignments.unsqueeze(1), 1)

    if buckets_to_use is not None:
        one_hot = one_hot[:, buckets_to_use]

    # one_hot *= x[:, None]
    
    return one_hot


def fraction_variance_unexplained(original, predicted):
    return ((original - predicted)**2).sum() / ((original - original.mean(dim=0))**2).sum()