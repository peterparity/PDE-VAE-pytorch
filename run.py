#!/bin/sh
''''exec python -u -- "$0" "$@" # '''
# vi: syntax=python

"""
run.py

Main script for training or evaluating a PDE-VAE model specified by the input file (JSON format).

Usage:
python run.py input_file.json > out
"""

import os
import sys
from shutil import copy2
import json
from types import SimpleNamespace
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

def setup(in_file):
    # Load configuration from json
    with open(in_file) as f:
        s = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    # Some defaults
    if not hasattr(s, 'train'):
        raise NameError("'train' must be set to True for training or False for evaluation.")
    elif s.train == False and not hasattr(s, 'MODELLOAD'):
        raise NameError("'MODELLOAD' file name required for evaluation.")

    if not hasattr(s, 'restart'):
        s.restart = not s.train
        warnings.warn("Automatically setting 'restart' to " + str(s.restart))
    if s.restart and not hasattr(s, 'MODELLOAD'):
        raise NameError("'MODELLOAD' file name required for restart.")

    if not hasattr(s, 'freeze_encoder'):
        s.freeze_encoder = False
    elif s.freeze_encoder and not s.restart:
        raise ValueError("Freeezing encoder weights requires 'restart' set to True with encoder weights loaded from file.")

    if not hasattr(s, 'data_parallel'):
        s.data_parallel = False
    if not hasattr(s, 'debug'):
        s.debug = False
    if not hasattr(s, 'discount_rate'):
        s.discount_rate = 0.
    if not hasattr(s, 'rate_decay'):
        s.rate_decay = 0.
    if not hasattr(s, 'param_dropout_prob'):
        s.param_dropout_prob = 0.
    if not hasattr(s, 'prop_noise'):
        s.prop_noise = 0.

    if not hasattr(s, 'boundary_cond'):
        raise NameError("Boundary conditions 'boundary_cond' not set. Options include: 'crop', 'periodic', 'dirichlet0'")
    elif s.boundary_cond == 'crop' and (not hasattr(s, 'input_size') or not hasattr(s, 'training_size')):
        raise NameError("'input_size' or 'training_size' not set for crop boundary conditions.")

    # Create output folder
    if not os.path.exists(s.OUTFOLDER):
        print("Creating output folder: " + s.OUTFOLDER)
        os.makedirs(s.OUTFOLDER)
    elif s.train and os.listdir(s.OUTFOLDER):
        raise FileExistsError("Output folder " + s.OUTFOLDER + " is not empty.")

    # Make a copy of the configuration file in the output folder
    copy2(in_file, s.OUTFOLDER)

    # Print configuration
    print(s)

    # Import class for dataset type
    dataset = __import__(s.dataset_type, globals(), locals(), ['PDEDataset'])
    s.PDEDataset = dataset.PDEDataset

    # Import selected model from models as PDEModel
    models = __import__('models.' + s.model, globals(), locals(), ['PDEAutoEncoder'])
    PDEModel = models.PDEAutoEncoder

    # Initialize model
    model = PDEModel(param_size=s.param_size, data_channels=s.data_channels, data_dimension=s.data_dimension,
                    hidden_channels=s.hidden_channels, linear_kernel_size=s.linear_kernel_size, 
                    nonlin_kernel_size=s.nonlin_kernel_size, prop_layers=s.prop_layers, prop_noise=s.prop_noise,
                    boundary_cond=s.boundary_cond, param_dropout_prob=s.param_dropout_prob, debug=s.debug)
    
    # Set CUDA device
    s.use_cuda = torch.cuda.is_available()
    if s.use_cuda:
        print("Using cuda device(s): " + str(s.cuda_device))
        torch.cuda.set_device(s.cuda_device)
        model.cuda()
    else:
        warnings.warn("Warning: Using CPU only. This is untested.")

    print("\nModel parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("\t{:<40}{}".format(name + ":", param.shape))

    return model, s


def _periodic_pad_1d(x, dim, pad):
    back_padding = x.narrow(dim, 0, pad)
    return torch.cat((x, back_padding), dim=dim)


def _random_crop_1d(sample, depth, crop_size):
    sample_size = sample[0].shape
    crop_t = [np.random.randint(sample_size[-2]-depth[0]+1), np.random.randint(sample_size[-2]-depth[1]+1)]
    crop_x = [np.random.randint(sample_size[-1]), np.random.randint(sample_size[-1])]
   
    if crop_size[0] > 1: 
        sample[0] = _periodic_pad_1d(sample[0], -1, crop_size[0]-1)
    if crop_size[1] > 1:
        sample[1] = _periodic_pad_1d(sample[1], -1, crop_size[1]-1)

    if len(sample_size) == 3:
        sample[0] = sample[0][:, crop_t[0]:(crop_t[0]+depth[0]), crop_x[0]:(crop_x[0]+crop_size[0])]
        sample[1] = sample[1][:, crop_t[1]:(crop_t[1]+depth[1]), crop_x[1]:(crop_x[1]+crop_size[1])]
    elif len(sample_size) == 2:
        sample[0] = sample[0][crop_t[0]:(crop_t[0]+depth[0]), crop_x[0]:(crop_x[0]+crop_size[0])]
        sample[1] = sample[1][crop_t[1]:(crop_t[1]+depth[1]), crop_x[1]:(crop_x[1]+crop_size[1])]
    else:
        raise ValueError('Sample is the wrong shape.')
        
    return sample


def _random_crop_2d(sample, depth, crop_size):
    sample_size = sample[0].shape
    crop_t = [np.random.randint(sample_size[-3]-depth[0]+1), np.random.randint(sample_size[-3]-depth[1]+1)]
    crop_x = [np.random.randint(sample_size[-2]), np.random.randint(sample_size[-2])]
    crop_y = [np.random.randint(sample_size[-1]), np.random.randint(sample_size[-1])]
    
    if crop_size[0] > 1:
        sample[0] = _periodic_pad_1d(_periodic_pad_1d(sample[0], -1, crop_size[0]-1), -2, crop_size[0]-1)
    if crop_size[1] > 1:
        sample[1] = _periodic_pad_1d(_periodic_pad_1d(sample[1], -1, crop_size[1]-1), -2, crop_size[1]-1)

    if len(sample_size) == 4:
        sample[0] = sample[0][:, crop_t[0]:(crop_t[0]+depth[0]), crop_x[0]:(crop_x[0]+crop_size[0]), crop_y[0]:(crop_y[0]+crop_size[0])]
        sample[1] = sample[1][:, crop_t[1]:(crop_t[1]+depth[1]), crop_x[1]:(crop_x[1]+crop_size[1]), crop_y[1]:(crop_y[1]+crop_size[1])]
    elif len(sample_size) == 3:
        sample[0] = sample[0][crop_t[0]:(crop_t[0]+depth[0]), crop_x[0]:(crop_x[0]+crop_size[0]), crop_y[0]:(crop_y[0]+crop_size[0])]
        sample[1] = sample[1][crop_t[1]:(crop_t[1]+depth[1]), crop_x[1]:(crop_x[1]+crop_size[1]), crop_y[1]:(crop_y[1]+crop_size[1])]
    else:
        raise ValueError('Sample is the wrong shape.')

    return sample


def train(model, s):
    ### Train model on training set
    print("\nTraining...")

    if s.restart: # load model to restart training
        print("Loading model from: " + s.MODELLOAD)
        strict_load = not s.freeze_encoder
        if s.use_cuda:
            state_dict = torch.load(s.MODELLOAD, map_location=torch.device('cuda', torch.cuda.current_device()))
        else:
            state_dict = torch.load(s.MODELLOAD)
        model.load_state_dict(state_dict, strict=strict_load)

        if s.freeze_encoder: # freeze encoder weights
            print("Freezing weights:")
            for name, param in model.encoder.named_parameters():
                param.requires_grad = False
                print("\t{:<40}{}".format("encoder." + name + ":", param.size()))
            for name, param in model.encoder_to_param.named_parameters():
                param.requires_grad = False
                print("\t{:<40}{}".format("encoder_to_param." + name + ":", param.size()))
            for name, param in model.encoder_to_logvar.named_parameters():
                param.requires_grad = False
                print("\t{:<40}{}".format("encoder_to_logvar." + name + ":", param.size()))

    if s.data_parallel:
        model = nn.DataParallel(model, device_ids=s.cuda_device)

    if s.boundary_cond == 'crop':
        if s.data_dimension == 1:
            transform = lambda x: _random_crop_1d(x, (s.input_depth, s.training_depth+1), (s.input_size, s.training_size)) 
        elif s.data_dimension == 2:
            transform = lambda x: _random_crop_2d(x, (s.input_depth, s.training_depth+1), (s.input_size, s.training_size)) 
        
        pad = int((2+s.prop_layers)*(s.nonlin_kernel_size-1)/2) #for cropping targets

    elif s.boundary_cond == 'periodic' or s.boundary_cond == 'dirichlet0':
        transform = None

    else:
        raise ValueError("Invalid boundary condition.")

    train_loader = torch.utils.data.DataLoader(
        s.PDEDataset(data_file=s.DATAFILE, transform=transform),
        batch_size=s.batch_size, shuffle=True, num_workers=s.num_workers, pin_memory=True,
        worker_init_fn=lambda _: np.random.seed())

    optimizer = torch.optim.Adam(model.parameters(), lr=s.learning_rate, eps=s.eps)

    model.train()

    writer = SummaryWriter(log_dir=os.path.join(s.OUTFOLDER, 'data'))

    # Initialize training variables
    loss_list = []
    recon_loss_list = []
    mse_list = []
    acc_loss = 0
    acc_recon_loss = 0
    acc_latent_loss = 0
    acc_mse = 0
    best_mse = None
    step = 0
    current_discount_rate = s.discount_rate

    ### Training loop
    for epoch in range(1, s.max_epochs+1):
        print('\nEpoch: ' + str(epoch))

        # Introduce a discount rate to favor predicting better in the near future
        current_discount_rate = s.discount_rate * np.exp(-s.rate_decay * (epoch-1)) # discount rate decay every epoch
        print('discount rate = ' + str(current_discount_rate))
        if current_discount_rate > 0:
            w = torch.tensor(np.exp(-current_discount_rate * np.arange(s.training_depth)).reshape(
                    [s.training_depth] + s.data_dimension * [1]), dtype=torch.float32, device='cuda' if s.use_cuda else 'cpu')
            w = w * s.training_depth/w.sum(dim=0, keepdim=True)
        else:
            w = None

        # Load batch and train
        for data, target, data_params in train_loader:
            step += 1

            if s.use_cuda:
                data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            data = data[:,:,:s.input_depth]
            if s.boundary_cond == 'crop':
                target0 = target[:,:,:s.training_depth]
                if s.data_dimension == 1:
                    target = target[:,:,1:s.training_depth+1, pad:-pad]
                elif s.data_dimension == 2:
                    target = target[:,:,1:s.training_depth+1, pad:-pad, pad:-pad]
            
            elif s.boundary_cond == 'periodic' or s.boundary_cond == 'dirichlet0':
                target0 = target[:,:,0]
                target = target[:,:,1:s.training_depth+1]

            else:
                raise ValueError("Invalid boundary condition.")

            # Run model
            output, params, logvar = model(data, target0, depth=s.training_depth)

            # Reset gradients
            optimizer.zero_grad()

            # Calculate loss
            if s.data_parallel:
                output = output.cpu()
            recon_loss = F.mse_loss(output * w, target * w) if w is not None else F.mse_loss(output, target)
            if s.param_size > 0:
                latent_loss = s.beta * 0.5 * torch.mean(torch.sum(params * params + logvar.exp() - logvar - 1, dim=-1))
            else:
                latent_loss = 0
            loss = recon_loss + latent_loss

            mse = F.mse_loss(output.detach(), target.detach()).item() if w is not None else recon_loss.item()

            loss_list.append(loss.item())
            recon_loss_list.append(recon_loss.item())
            mse_list.append(mse)

            acc_loss += loss.item()
            acc_recon_loss += recon_loss.item()
            acc_latent_loss += latent_loss.item()
            acc_mse += mse

            # Calculate gradients
            loss.backward()

            # Clip gradients
            # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1e0)

            # Update gradients
            optimizer.step()

            # Output every 100 steps
            if step % 100 == 0:
                # Check every 500 steps and save checkpoint if new model is at least 2% better than previous best
                if (step > 1 and step % 500 == 0) and ((best_mse is None) or (acc_mse/100 < 0.98*best_mse)):
                    best_mse = acc_mse/100
                    torch.save(model.state_dict(), os.path.join(s.OUTFOLDER, "best.tar"))
                    print('New Best MSE at Step {}: {:.4f}'.format(step, best_mse))

                # Output losses and weights
                if s.param_size > 0:
                    if step > 1:
                        # Write losses to summary
                        writer.add_scalars('losses',    {'loss': acc_loss/100,
                                                         'recon_loss': acc_recon_loss/100,
                                                         'latent_loss': acc_latent_loss/100,
                                                         'mse': acc_mse/100}, step)

                        acc_loss = 0
                        acc_recon_loss = 0
                        acc_latent_loss = 0
                        acc_mse = 0

                    # Write mean model weights to summary
                    weight_dict = {}
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            weight_dict[name] = param.detach().abs().mean().item()
                    writer.add_scalars('weight_avg', weight_dict, step)

                    print('Train Step: {}\tTotal Loss: {:.4f}\tRecon. Loss: {:.4f}\tRecon./Latent: {:.1f}\tMSE: {:.4f}'
                            .format(step, loss.item(), recon_loss.item(), recon_loss.item()/latent_loss.item(), mse))
                    
                    # Save current set of extracted latent parameters
                    np.savez(os.path.join(s.OUTFOLDER, "training_params.npz"),  data_params=data_params.numpy(), 
                                                                                params=params.detach().cpu().numpy())
                else:
                    print('Train Step: {}\tTotal Loss: {:.4f}\tRecon. Loss: {:.4f}\tMSE: {:.4f}'
                            .format(step, loss.item(), recon_loss.item(), mse))

        # Export checkpoints and loss history after every s.save_epochs epochs
        if s.save_epochs > 0 and epoch % s.save_epochs == 0:
            torch.save(model.state_dict(), os.path.join(s.OUTFOLDER, "epoch{:06d}.tar".format(epoch)))
            np.savez(os.path.join(s.OUTFOLDER, "loss.npz"), loss=np.array(loss_list), 
                                                            recon_loss=np.array(recon_loss_list), 
                                                            mse=np.array(mse_list))

    return model


def evaluate(model, s, params_filename="params.npz", rmse_filename="rmse_with_depth.npy"):
    ### Evaluate model on test set
    print("\nEvaluating...")

    if rmse_filename is not None and os.path.exists(os.path.join(s.OUTFOLDER, rmse_filename)):
        raise FileExistsError(rmse_filename + " already exists.")
    if os.path.exists(os.path.join(s.OUTFOLDER, params_filename)):
        raise FileExistsError(params_filename + " already exists.")

    if not s.train:
        print("Loading model from: " + s.MODELLOAD)
        if s.use_cuda:
            state_dict = torch.load(s.MODELLOAD, map_location=torch.device('cuda', torch.cuda.current_device()))
        else:
            state_dict = torch.load(s.MODELLOAD)
        model.load_state_dict(state_dict)
        
    pad = int((2+s.prop_layers)*(s.nonlin_kernel_size-1)/2) #for cropping targets (if necessary)

    test_loader = torch.utils.data.DataLoader(
        s.PDEDataset(data_file=s.DATAFILE, transform=None),
        batch_size=s.batch_size, num_workers=s.num_workers, pin_memory=True)

    model.eval()
    torch.set_grad_enabled(False)

    ### Evaluation loop
    loss = 0
    if rmse_filename is not None:
        rmse_with_depth = torch.zeros(s.evaluation_depth, device='cuda' if s.use_cuda else 'cpu')
    params_list = []
    logvar_list = []
    data_params_list = []
    step = 0
    for data, target, data_params in test_loader:
        step += 1

        if s.use_cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

        if s.boundary_cond == 'crop':
            target0 = target[:,:,:s.evaluation_depth]
            if s.data_dimension == 1:
                target = target[:,:,1:s.evaluation_depth+1, pad:-pad]
            elif s.data_dimension == 2:
                target = target[:,:,1:s.evaluation_depth+1, pad:-pad, pad:-pad]
        
        elif s.boundary_cond == 'periodic' or s.boundary_cond == 'dirichlet0':
            target0 = target[:,:,0]
            target = target[:,:,1:s.evaluation_depth+1]

        else:
            raise ValueError("Invalid boundary condition.")

        # Run model
        if s.debug:
            output, params, logvar, _, weights, raw_params = model(data.contiguous(), target0, depth=s.evaluation_depth)
        else:
            output, params, logvar = model(data.contiguous(), target0, depth=s.evaluation_depth)

        data_params = data_params.numpy()
        data_params_list.append(data_params)

        if s.param_size > 0:
            params = params.detach().cpu().numpy()
            params_list.append(params)
            logvar_list.append(logvar.detach().cpu().numpy())

        assert output.shape[2] == s.evaluation_depth
        loss += F.mse_loss(output, target).item()

        if rmse_filename is not None:
            rmse_with_depth += torch.sqrt(torch.mean((output - target).transpose(2,1).contiguous()
                                        .view(target.size()[0], s.evaluation_depth, -1) ** 2,
                                                 dim=-1)).mean(0)

    rmse_with_depth = rmse_with_depth.cpu().numpy()/step
    print('\nTest Set: Recon. Loss: {:.4f}'.format(loss/step))

    if rmse_filename is not None:
        np.save(os.path.join(s.OUTFOLDER, rmse_filename), rmse_with_depth)

    np.savez(os.path.join(s.OUTFOLDER, params_filename), params=np.concatenate(params_list), 
                                                         logvar=np.concatenate(logvar_list),
                                                         data_params=np.concatenate(data_params_list))


if __name__ == "__main__":
    in_file = sys.argv[1]
    if not os.path.exists(in_file):
        raise FileNotFoundError("Input file " + in_file + " not found.")

    model, s = setup(in_file)
    if s.train:
        model = train(model, s)
    else:
        evaluate(model, s)
