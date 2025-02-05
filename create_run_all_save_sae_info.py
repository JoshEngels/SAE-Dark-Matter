# %%
from utils import layer_to_sae_ids, gemma_sae_info_to_params, sae_info_to_params
import os
import matplotlib.pyplot as plt

os.makedirs("scripts", exist_ok=True)


num_gpus = 1

gpu_offset = 0

all_commands = []

# %%

# Single layer experiments
layer_type = "res"
for model_name, layers in [("gemma_2_2b", [5, 12, 19]), ("gemma_2_9b", [9, 20, 31])]:

    sae_ids = []
    for layer in layers:
        sae_ids.extend(layer_to_sae_ids(layer, model_name, layer_type))

    for sae_id in sae_ids:
        params = gemma_sae_info_to_params(sae_id)
        
        all_commands.append(
            f"python save_sae_info.py --layer {params['layer']} --sae_id {sae_id} --model_name {model_name} --layer_type {layer_type}"
        )
        

# %%

def get_sae_ids_closest_to_target_l0(model_name, layer_type, target_l0):
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

# Multi-layer experiments
layer_type = "res"
for model_name, widths, layers in [("gemma_2_2b", ["16k", "65k"], range(26)), ("gemma_2_9b", ["16k", "131k"], range(41))]:
    for width in widths:
        target_l0_to_average_diff = {}
        for target_l0 in range(10, 400, 10):            
            all_layer_sae_ids = get_sae_ids_closest_to_target_l0(model_name, layer_type, target_l0)
            l0s = [sae_info_to_params(sae_id, model_name)["l0"] for sae_id in all_layer_sae_ids]
            average_l0 = sum(l0s) / len(l0s)
            average_diff = sum([abs(l0 - average_l0) for l0 in l0s]) / len(l0s)
            target_l0_to_average_diff[target_l0] = average_diff

        target_l0s = sorted(target_l0_to_average_diff.keys())
        average_diffs = [target_l0_to_average_diff[l0] for l0 in target_l0s]
        
        plt.figure(figsize=(10,6))
        plt.plot(target_l0s, average_diffs)
        for x in range(10, 400, 10):
            plt.axvline(x=x, color='gray', linestyle='--', alpha=0.3)
        plt.xlabel("Target L0")
        plt.ylabel("Average L0 Difference")
        plt.title(f"Target L0 vs Average L0 Difference\n{model_name} width={width}")
        plt.show()


# %%

# From manual inspection of the plots above
model_name_and_width_to_target_l0s = {
    ("llama_3.1_8b", "8x"): [50],
    ("llama_3.1_8b", "32x"): [50],
    ("gemma_2_2b", "16k"): [20, 70, 160, 350],
    ("gemma_2_2b", "65k"): [20, 70, 120, 250],
    ("gemma_2_9b", "16k"): [10, 60, 120, 250],
    ("gemma_2_9b", "131k"): [10, 60, 100, 250]
}

# %%

for model_name, widths, layers in [("llama_3.1_8b", ["8x", "32x"], range(50)), ("gemma_2_2b", ["16k", "65k"], range(26)), ("gemma_2_9b", ["16k", "131k"], range(41))]:
    for width in widths:
        target_l0s = model_name_and_width_to_target_l0s[(model_name, width)]
        for target_l0 in target_l0s:
            all_layer_sae_ids = get_sae_ids_closest_to_target_l0(model_name, layer_type, target_l0)
            for sae_id in all_layer_sae_ids:
                params = sae_info_to_params(sae_id, model_name)
                all_commands.append(
                    f"python save_sae_info.py --layer {params['layer']} --sae_id {sae_id} --model_name {model_name} --layer_type {layer_type}"
                )
            
                
# %%

with open("scripts/run_all.sh", "w") as f:
    for i in range(0, len(all_commands), num_gpus):
        for device_id in range(num_gpus):
            if i + device_id < len(all_commands):
                f.write(f"{all_commands[i + device_id]} --device cuda:{device_id + gpu_offset} &\n")
        f.write("wait\n")