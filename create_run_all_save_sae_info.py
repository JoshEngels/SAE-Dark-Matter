# %%
from utils import layer_to_sae_ids, gemma_sae_info_to_params
import os

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

with open("scripts/run_all.sh", "w") as f:
    for i in range(0, len(all_commands), num_gpus):
        for device_id in range(num_gpus):
            if i + device_id < len(all_commands):
                f.write(f"{all_commands[i + device_id]} --device cuda:{device_id + gpu_offset} &\n")
        f.write("wait\n")