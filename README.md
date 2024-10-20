# Decomposing the Dark Matter of Sparse Autoencoders
This is the github repo for our paper "Decomposing the Dark Matter of Sparse Autoencoders".

Below are instructions explaining how to reproduce each figure from the paper. 

We also include instructions for generating and using SAEInfoObjects, a simple yet powerful python summary object which contains a wealth of information about a given SAE on a dataset, including model activations, SAE activations, SAE feature frequencies, SAE reconstructions, and SAE errors. We are excited to see what else people can learn about SAEs using this interface!

## Setting up the Environment

The required python packages to run this repo are
```
transformer_lens sae_lens transformers datasets torch nnsight aiofiles openai
```
We recommend you create a new python venv named e.g. darkmatter and install these packages manually using pip:
```
python -m venv darkmatter
source darkmatter/bin/activate
pip install transformer_lens sae_lens transformers datasets torch nnsight aiofiles openai
```

To run auto interp experiments, you will also need to run
```
git submodule update --init --recursive
cd sae-auto-interp
pip install -e .
```
to get the required auto interp submodule.

Let us know if anything does not work with this environment!

You will also need to add a directory to BASE_DIR in utils.py that can store a few hundred GB of data, as this will store model and SAE activations and info (see the next section).

## Generating Activations

To run our experiments, you will need to generate and save model activations for all layers of interest on our token evaluation set, as well as SAE activations and metadata for all SAEs of interest.

To generating and save model activations, run
```
python3 save_gemma_acts_and_loss.py --device cuda:0 --layer_type res
python3 save_gemma_acts_and_loss.py --device cuda:0 --layer_type mlp --layers 19 20
python3 save_gemma_acts_and_loss.py --device cuda:0 --layer_type att --layers 19 20
```

To generate and save sae information, run
```
python3 scripts/create_run_all_for_gemma_single_layer.py --num_gpus 1 --device_offset 0 
./scripts/run_all.sh
python3 scripts/create_run_all_for_gemma_all_layers.py --num_gpus 1 --device_offset 0
./scripts/run_all.sh
python3 scripts/create_run_all_for_pursuit.py --device cuda:0
./scripts/run_all.sh
```

## The SAEInfoObject

The above code saves SAE activations and other info in BASE_DIR that will now be loadable into an SAEInfoObject. This is a convenient interface for performing many types of analysis on SAEs, and so we include a tutorial for how to use it in sae_info_object_tutorial.py. 

## Reproducing Figures

For reproducing Figure 1, Figure 2, Figure 3, Figure 4, Figure 5, Figure 6, Figure 10a, Figure 11, and Figure 12, run
```
python3 sae_power_laws.py --device cuda:0 --to_plot both
python3 sae_power_laws.py --device cuda:0 --to_plot pursuit
python3 sae_power_laws_per_token.py --device cuda:0
```

Generating Figure 8 is more involved. First, train SAEs on each component of the error:
```
python3 train_on_error.py --device cuda:0 --train_on linear_prediction_of_error
python3 train_on_error.py --device cuda:0 --train_on nonlinear_error
```
Then, 


Plots will be saved to the plots/ directory.

## Contact

If you have any questions about the paper or reproducing results, feel free to email jengels@mit.edu.

## Note about SAE training implementation
For training topk SAEs we use a slightly modified version of https://github.com/EleutherAI/sae/tree/main stored in eleuther_sae_modified.