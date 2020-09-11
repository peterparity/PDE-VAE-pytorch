# PDE-VAE-pytorch

Implementation of a variational autoencoder (VAE)-based method for extracting interpretable physical parameters (from spatiotemporal data) that parameterize the dynamics of a spatiotemporal system, e.g. a system governed by a partial differential equation (PDE).

Please cite "**Extracting Interpretable Physical Parameters from Spatiotemporal Systems using Unsupervised Learning**" (https://journals.aps.org/prx/abstract/10.1103/PhysRevX.10.031056) and see the paper for more details. This is the official repository for the paper.

## Requirements
PyTorch version >= 1.1.0, NumPy

(Note: Dataset generation scripts have additional requirements is some cases.)

## Usage
### Dataset
The dataset generation scripts for the datasets in the paper are located in the "data/" folder. Data is loaded using the PyTorch dataloader framework. To use the existing dataset loader, format the data as a NumPy array with shape:

```
(dataset size, data channels, propagation dimension, spatial dimension 1, spatial dimension 2, ...)
```
Currently, only datasets with 1 or 2 spatial dimensions are supported. The propagation dimension is usually the time direction.

### Training
Hyperparameter and architecture adjustments can be made using the input file. Examples are located in the "input\_files/" folder (see "input\_files/template.json" for a description of each setting). For training, make sure the "train" parameter is set to *true* in the input file, then run:

```bash
python run.py input_file.json > out
```

### Evaluation
To run the provided evaluation script, change the "train" parameter to *false* in the input file, and make sure to set "MODELLOAD" in the input file to the path of the trained model save. Then, rerun the same input file. Note that even if crop boundaries are used, the evaluation method will no longer crop to smaller sizes and instead evaluates on the full dataset, so adjustments may need to be made to the boundary conditions and batch size.

Custom evaluation routines are recommended for detailed data analysis. For example, using the "pde1d_decoder_only" model (or "pde2d_decoder_only" model) with weights loaded from the trained model allows you to manually tune the latent parameters and observe the predicted propagation (given an input initial condition). This may aid in interpreting the extracted relevant parameters.
