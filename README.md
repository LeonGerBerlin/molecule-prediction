**Work in progress**

### Molecular prediction toolkit

Simple molecule prediction tool for regression tasks:

- Uses a solubility dataset for model optimization / testing.
- Implements a MPNN framework with attention or graph convolution layers.
- As baseline LightGBM is available.
- For hyperparameter sweeps one can use W&B (see function `sweep_wb` in `utils/wandb_utils.py`).

### Getting started

1. Install all required packages (see `requirements.txt`).
2. Create W&B account for logging training data.
3. Log into W&B. 
4. Call `main.py`. Fix data path.  

### Datasets

Solubility dataset, Fang et al.,
"Prospective Validation of Machine Learning Algorithms for Absorption, Distribution, Metabolism, and Excretion Prediction: An Industrial Perspective", 2023.


### Credits

Input features
- [DiffDock](https://github.com/gcorso/DiffDock) 

Dataset
- Fang et al. (2023).
- https://practicalcheminformatics.blogspot.com/2023/06/getting-real-with-molecular-property.html
