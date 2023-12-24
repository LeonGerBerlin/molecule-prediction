**Work in progress**

## Molecular property prediction framework
A python package to perform molecular property prediction with pre-trained regression models.

Free software: MIT license

Installation
--------

1. Setup an environment either using conda or virtualenv.
The code is tested so far with python 3.10.

2. Install the dependencies from the `requirements.txt` file,
e.g. `pip install -r requirements.txt`. Additionally, install manually
lightgbm (due to problems with M2 MacBook).

3. Install the python package: `pip install -e .`

4. Connect to your W&B account.


For M2 MacBook to install lightgbm:
```bash
conda install --yes -c conda-forge 'lightgbm>=3.3.3'
```

Getting started
--------
For help about the command line flags:
```bash
molproperty_prediction -h
```

### Optimizing machine learning model
The code currently assumes a .csv for the dataset with `smiles` and `label` column name.

#### Single training run
Setup a training run with a .csv dataset:
```bash
 molproperty_prediction -m train -f "data/biogen_solubility.csv" --config-path "configs/config_mpnn.yml"
 ```

#### W&B Hyperparameter sweep
To perform a hyperparameter sweep:
```bash
molproperty_prediction -m wb_sweep -f "data/biogen_solubility.csv" --wb-config-path "configs/config_wb.yml" --config-path "configs/config_mpnn.yaml"
```
In case you don't define the `wb-config-path` a default hyperparameter grid is used. Hyperparameter sweeps currently only work for GNN architectures and not
machine learning models like `lgbm`.

### Inference
To use a pre-trained model call the package using the inference mode:
```bash
molproperty_prediction -m inference -f "data/biogen_solubility.csv" --checkpoint-path "checkpoints/model_checkpoint_ep_66_loss_0.33.pt"
```
The code file dump a .csv file with SMILES strings and predicted labels.


Features
--------
*Disclaimer*: Not all features are rigorously tested.

- Optimization of a GNN & LGBM machine learning model for a regression task using SMILES strings as input and prediction of a one dimensional regression value.
- W&B logging: All runs get uploaded to your W&B project.
- W&B Sweep: Hyperparameter sweep for a GNN.

To Do's
--------
- Sphinx documentation
- Pre-training for neural network approaches
- Tests...

Credits
-------

As a starting point a Cookiecutter template was used:
- Cookiecutter: https://github.com/audreyr/cookiecutter
- `audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

Input features:
- [DiffDock](https://github.com/gcorso/DiffDock)

Dataset:
- Solubility dataset, Fang et al., "Prospective Validation of Machine Learning Algorithms for Absorption, Distribution, Metabolism, and Excretion Prediction: An Industrial Perspective", 2023.
- https://practicalcheminformatics.blogspot.com/2023/06/getting-real-with-molecular-property.html



