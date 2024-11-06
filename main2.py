#!/usr/bin/env python
# coding: utf-8

# Import header files
import argparse
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

from models.esru_1LF import eSRU_1LF, train_eSRU_1LF
from models.esru_2LF import eSRU_2LF, train_eSRU_2LF
from models.sru import SRU, trainSRU
from utils.utilFuncs import env_config, count_parameters, \
    getGeneTrainingData

# Read input command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='device, default: cpu')
parser.add_argument('--dataset', type=str, default='VAR', help='dataset type, default: VAR')
parser.add_argument('--dsid', type=int, default=1, help='dataset id, default: 1')
parser.add_argument('--T', type=int, default=10, help='training size, default: 10')
parser.add_argument('--F', type=int, default=10, help='chaos, default: 10')
parser.add_argument('--sruname', type=int, default=10, help='num of timeseries, default: 10')
parser.add_argument('--model', type=str, default='sru', help='[sru, gru, lstm]: select your model')
parser.add_argument('--nepochs', type=int, default=500, help='sets max_iter, default: 500')
parser.add_argument('--mu1', type=float, default=1, help='sets mu1 parameter, default: 1')
parser.add_argument('--mu2', type=float, default=1, help='sets mu2 parameter, default: 1')
parser.add_argument('--mu3', type=float, default=1, help='sets mu3 parameter, default: 1')
parser.add_argument('--lr', type=float, default=0.005, help='sets learning rate, default: 0.005')
parser.add_argument('--joblog', type=str, default="", help='name of job logfile, default=""')
parser.add_argument('--inputfile', type=str, default="", help='path of input file, default=""')

args = parser.parse_args()
deviceName = args.device
model_name = args.model
max_iter = args.nepochs
mu1 = args.mu1
mu2 = args.mu2
mu3 = args.mu3
dataset = args.dataset
dataset_id = args.dsid
T = args.T
F = args.F
n = args.sruname
lr = args.lr
jobLogFilename = args.joblog
fileName = args.inputfile

###############################
# Global simulation settings
###############################
verbose = 0  # Verbosity level

#################################
# Pytorch environment
#################################
device, seed = env_config(True, deviceName)  # true --> use GPU
print("Computational Resource: %s" % (device))

######################################
# Create input data in batch format
######################################

if (dataset == 'lorenz'):
    # Load data
    ld = np.load(fileName)
    X_np = ld['X_np']
    Gref = ld['Gref']
    numTotalSamples = T
    Xtrain = torch.from_numpy(X_np)
    Xtrain = Xtrain.float().to(device)
    inputSignalMultiplier = 1
    Xtrain = inputSignalMultiplier * Xtrain

elif (dataset == 'var'):
    # Load data for VAR dataset
    ld = np.load(fileName)
    X_np = ld['X_np']
    Gref = ld['Gref']
    numTotalSamples = T
    Xtrain = torch.from_numpy(X_np)
    Xtrain = Xtrain.float().to(device)
    inputSignalMultiplier = 1
    Xtrain = inputSignalMultiplier * Xtrain

else:
    print("Unsupported dataset encountered")

##################################
# SRU Cell parameters
##################################
# Add more custom model parameters here if needed.

##################################
# Model training parameters
##################################

# Model setup for SRU (or other models)
if model_name == 'sru':
    lr_gamma = 0.99
    lr_update_gap = 4
    stoppingThresh = 1e-5
    trainVerboseLvl = 2
    lambda1 = mu1
    lambda2 = mu2
    n_inp_channels = n
    n_out_channels = 1
    batchSize = 250
    blk_size = int(batchSize / 2)
    numBatches = int(numTotalSamples / batchSize)

    model = SRU(n_inp_channels, n_out_channels, 10, 10, 10, 10, [0.0, 0.01, 0.1, 0.99], device)
    model.to(device)
    print(count_parameters(model))
    model, lossVec = trainSRU(model, Xtrain, device, numBatches, batchSize, blk_size, 0, max_iter,
                              lambda1, lambda2, lr, lr_gamma, lr_update_gap, 1, stoppingThresh, trainVerboseLvl)

    Gest = torch.norm(model.lin_xr2phi.weight.data[:, :n], p=2, dim=0)

elif model_name == 'eSRU_1LF':
    # Similar setup for eSRU_1LF model
    pass
elif model_name == 'eSRU_2LF':
    # Similar setup for eSRU_2LF model
    pass
else:
    print("Unsupported model encountered")

###########################
# Compute AUROC
###########################

# Assuming Gest contains predicted values and Gref contains true labels
# Flatten the outputs for AUROC calculation
Gest_flat = Gest.detach().cpu().numpy().flatten()
Gref_flat = Gref.flatten()

# Compute AUROC score
try:
    auroc = roc_auc_score(Gref_flat, Gest_flat)
    print(f"AUROC: {auroc}")
except ValueError as e:
    print(f"Error computing AUROC: {e}")

##############################
# Save job log
##############################

if jobLogFilename != "":
    np.savez(jobLogFilename,
             Gref=Gref,
             Gest=Gest.detach().cpu().numpy(),
             model=model_name,
             dataset=dataset,
             dsid=dataset_id,
             T=T,
             F=F,
             nepochs=max_iter,
             mu1=mu1,
             mu2=mu2,
             mu3=mu3,
             lr=lr,
             batchSize=batchSize,
             blk_size=blk_size,
             numBatches=numBatches,
             dim_iid_stats=10,
             dim_rec_stats=10,
             dim_final_stats=10,
             dim_rec_stats_feedback=10)

# sleep for one second followed by printing
# the exit key for tmux consumption
time.sleep(1)
print("#RUN_COMPLETE #RUN_COMPLETE #RUN_COMPLETE #RUN_COMPLETE")
