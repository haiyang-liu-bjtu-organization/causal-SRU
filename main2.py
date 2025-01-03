#!/usr/bin/env python
# coding: utf-8

# Import header files
import argparse
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from models.esru_1LF import eSRU_1LF, train_eSRU_1LF
from models.esru_2LF import eSRU_2LF, train_eSRU_2LF
from models.sru import SRU, trainSRU
from utils.utilFuncs import env_config, count_parameters, \
    getGeneTrainingData

# Read input command line arguments
parser = argparse.ArgumentParser()
# parser.add_argument('--device', type=str, default='cuda:0',
#                      help='device, default: cuda:0')
# 添加 --device 参数
parser.add_argument('--device', type=str, default='cpu',  # 默认值改为 'cpu'
                    help='device, default: cpu')
parser.add_argument('--dataset', type=str, default='VAR',
                    help='dataset type, default: VAR')
parser.add_argument('--dsid', type=int, default=1,
                    help='dataset id, default: 1')
parser.add_argument('--T', type=int, default=10,
                    help='training size, default: 10')
parser.add_argument('--F', type=int, default=10,
                    help='chaos, default: 10')
parser.add_argument('--sruname', type=int, default=10,
                    help='num of timeseries, default: 10')
parser.add_argument('--model', type=str, default='sru',
                    help='[sru, gru, lstm]: select your model')
parser.add_argument('--nepochs', type=int, default=500,
                    help='sets max_iter, default: 500')
parser.add_argument('--mu1', type=float, default=1,
                    help='sets mu1 parameter, default: 1')
parser.add_argument('--mu2', type=float, default=1,
                    help='sets mu2 parameter, default: 1')
parser.add_argument('--mu3', type=float, default=1,
                    help='sets mu3 parameter, default: 1')
parser.add_argument('--lr', type=float, default=0.005,
                    help='sets learning rate, default: 0.005')
parser.add_argument('--joblog', type=str, default="",
                    help='name of job logfile, default=""')
# 新加的
parser.add_argument('--inputfile', type=str, default="",
                    help='path of input file, default=""')

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
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Computational Resource: %s" % (device))

######################################
# Create input data in batch format
######################################

if (dataset == 'gene'):
    Xtrain, Gref = getGeneTrainingData(dataset_id, device)
    n1 = Xtrain.shape[0]
    if (n != n1):
        print("Error::Dimension mismatch for input training data..")
    numTotalSamples = Xtrain.shape[1]
    Xtrain = Xtrain.float().to(device)
    # Make input signal zero mean and appropriately scaled
    Xtrain = Xtrain - Xtrain.mean()
    inputSignalMultiplier = 50
    Xtrain = inputSignalMultiplier * Xtrain

elif (dataset == 'var'):
    # fileName = "data/var/S_%s_T_%s_dataset_%s.npz" % (F, T, dataset_id)
    ld = np.load(fileName)
    X_np = ld['X']
    X_np = X_np.T  # 咱们的数据是(200,10)这种形式，然后咱们要转置成(10,200)
    Gref = ld['GC']
    numTotalSamples = T
    Xtrain = torch.from_numpy(X_np)
    Xtrain = Xtrain.float().to(device)
    inputSignalMultiplier = 1
    Xtrain = inputSignalMultiplier * Xtrain
    print(Xtrain)
    print("shape:", Xtrain.shape)

elif (dataset == 'lorenz'):
    # fileName = "data/lorenz96/F_%s_T_%s_dataset_%s.npz" % (F, T, dataset_id)
    # fileName = input
    ld = np.load(fileName)
    X_np = ld['X']
    X_np = X_np.T  # 咱们的数据是(200,10)这种形式，然后咱们要转置成(10,200)
    Gref = ld['GC']
    numTotalSamples = T
    Xtrain = torch.from_numpy(X_np)
    Xtrain = Xtrain.float().to(device)
    inputSignalMultiplier = 1
    Xtrain = inputSignalMultiplier * Xtrain
    print(Xtrain)
    print("shape:", Xtrain.shape)

elif (dataset == 'netsim'):
    # fileName = "data/netsim/sim3_subject_%s.npz" % (dataset_id)
    ld = np.load(fileName)
    X_np = ld['X_np']
    Gref = ld['Gref']
    numTotalSamples = T
    Xtrain = torch.from_numpy(X_np)
    Xtrain = Xtrain.float().to(device)
    inputSignalMultiplier = 1
    Xtrain = inputSignalMultiplier * Xtrain

else:
    print("Dataset is not supported")

if (verbose >= 1):
    plt.figure(1)
    plt.xlabel("t")
    plt.ylabel("x0(t)")
    plt.plot(range(numTotalSamples), Xtrain.cpu().numpy()[0][:])
    plt.show(block=False)
    plt.pause(0.1)

######################################
# SRU Cell parameters
######################################


#######################################
# Model training parameters
######################################
if (model_name == 'sru'):

    lr_gamma = 0.99
    lr_update_gap = 4
    staggerTrainWin = 1
    stoppingThresh = 1e-5;
    trainVerboseLvl = 2
    lr = lr
    lambda1 = mu1
    lambda2 = mu2
    n_inp_channels = n
    n_out_channels = 1

    if (dataset == 'gene'):
        A = [0.0, 0.01, 0.1, 0.5, 0.99];  # 0.75
        dim_iid_stats = 10  # math.ceil(n) #1.5n
        dim_rec_stats = 10  # math.ceil(n) #1.5n
        dim_final_stats = 10  # d * len(A) #math.ceil(n/2)
        dim_rec_stats_feedback = 10  # d * len(A)
        batchSize = 21
        blk_size = batchSize
        numBatches = int(numTotalSamples / batchSize)


    elif (dataset == 'var'):
        A = [0.0, 0.01, 0.1, 0.99];
        dim_iid_stats = 10  # math.ceil(n) #1.5n
        dim_rec_stats = 10  # math.ceil(n) #1.5n
        dim_final_stats = 10  # d * len(A) #math.ceil(n/2) #n
        dim_rec_stats_feedback = 10  # d * len(A) #math.ceil(n/2) #n
        batchSize = T
        blk_size = int(batchSize / 2)
        numBatches = int(numTotalSamples / batchSize)


    elif (dataset == 'lorenz'):
        A = [0.0, 0.01, 0.1, 0.99];
        dim_iid_stats = 10
        dim_rec_stats = 10
        dim_final_stats = 10
        dim_rec_stats_feedback = 10
        # batchSize = 250
        # batchSize = 200
        # batchSize = 500
        batchSize = T
        blk_size = int(batchSize / 2)
        numBatches = int(numTotalSamples / batchSize)

    elif (dataset == 'netsim'):
        A = [0.0, 0.01, 0.05, 0.1, 0.99];
        dim_iid_stats = 10
        dim_rec_stats = 10
        dim_final_stats = 10
        dim_rec_stats_feedback = 10
        batchSize = 10  # 100
        blk_size = int(batchSize / 2)
        numBatches = int(numTotalSamples / batchSize)

    else:
        print("Unsupported dataset encountered")

elif (model_name == 'eSRU_1LF' or model_name == 'eSRU_2LF'):

    lr_gamma = 0.99
    lr_update_gap = 4
    staggerTrainWin = 1
    stoppingThresh = 1e-5;
    trainVerboseLvl = 2
    lr = lr
    lambda1 = mu1
    lambda2 = mu2
    lambda3 = mu3
    n_inp_channels = n
    n_out_channels = 1

    if (dataset == 'gene'):
        A = [0.05, 0.1, 0.2, 0.99];
        dim_iid_stats = 10
        dim_rec_stats = 10
        dim_final_stats = 10
        dim_rec_stats_feedback = 10
        batchSize = 21
        blk_size = int(batchSize)
        numBatches = int(numTotalSamples / batchSize)


    elif (dataset == 'var'):
        A = [0.0, 0.01, 0.1, 0.99];
        dim_iid_stats = 10  # math.ceil(n) #1.5n
        dim_rec_stats = 10  # math.ceil(n) #1.5n
        dim_final_stats = 10  # d * len(A) #math.ceil(n/2) #n
        dim_rec_stats_feedback = 10  # d * len(A) #math.ceil(n/2) #n
        batchSize = 250
        blk_size = int(batchSize / 2)
        numBatches = int(numTotalSamples / batchSize)


    elif (dataset == 'lorenz'):
        # lr = 0.01
        A = [0.0, 0.01, 0.1, 0.99];
        dim_iid_stats = 10
        dim_rec_stats = 10
        dim_final_stats = 10  # d*len(A)
        dim_rec_stats_feedback = 10  # d*len(A)
        batchSize = 250
        blk_size = int(batchSize / 2)
        numBatches = int(numTotalSamples / batchSize)

    elif (dataset == 'netsim'):
        A = [0.0, 0.01, 0.1, 0.99];
        dim_iid_stats = 10
        dim_rec_stats = 10
        dim_final_stats = 10  # d*len(A)
        dim_rec_stats_feedback = 10  # d*len(A)
        batchSize = 10  # 10 #100
        blk_size = int(batchSize / 2)
        numBatches = int(numTotalSamples / batchSize)

    else:
        print("Unsupported dataset encountered")

else:
    print("Unsupported model encountered")

############################################
# Evaluate ROC plots (regress mu2)
############################################
if 1:
    Gest = torch.zeros(n, n, requires_grad=False)

    if (model_name == 'sru'):
        for predictedNode in range(n):
            start = time.time()
            print("node = %d" % (predictedNode))
            model = SRU(n_inp_channels, n_out_channels, dim_iid_stats, dim_rec_stats, dim_rec_stats_feedback,
                        dim_final_stats, A, device)
            model.to(device)  # shift to CPU/GPU memory
            print(count_parameters(model))
            model, lossVec = trainSRU(model, Xtrain, device, numBatches, batchSize, blk_size, predictedNode, max_iter,
                                      lambda1, lambda2, lr, lr_gamma, lr_update_gap, staggerTrainWin, stoppingThresh,
                                      trainVerboseLvl)
            Gest.data[predictedNode, :] = torch.norm(model.lin_xr2phi.weight.data[:, :n], p=2, dim=0)
            # print("Elapsed time (1) = % s seconds" % (time.time() - start))

    elif (model_name == 'eSRU_1LF'):
        for predictedNode in range(n):
            start = time.time()
            print("node = %d" % (predictedNode))
            model = eSRU_1LF(n_inp_channels, n_out_channels, dim_iid_stats, dim_rec_stats, dim_rec_stats_feedback,
                             dim_final_stats, A, device)
            model.to(device)  # shift to CPU/GPU memory
            print(count_parameters(model))
            model, lossVec = train_eSRU_1LF(model, Xtrain, device, numBatches, batchSize, blk_size, predictedNode,
                                            max_iter,
                                            lambda1, lambda2, lambda3, lr, lr_gamma, lr_update_gap, staggerTrainWin,
                                            stoppingThresh, trainVerboseLvl)
            Gest.data[predictedNode, :] = torch.norm(model.lin_xr2phi.weight.data[:, :n], p=2, dim=0)
            # print("Elapsed time (1) = % s seconds" % (time.time() - start))

    elif (model_name == 'eSRU_2LF'):
        for predictedNode in range(n):
            start = time.time()
            print("node = %d" % (predictedNode))
            model = eSRU_2LF(n_inp_channels, n_out_channels, dim_iid_stats, dim_rec_stats, dim_rec_stats_feedback,
                             dim_final_stats, A, device)
            model.to(device)  # shift to CPU/GPU memory
            print(count_parameters(model))
            model, lossVec = train_eSRU_2LF(model, Xtrain, device, numBatches, batchSize, blk_size, predictedNode,
                                            max_iter,
                                            lambda1, lambda2, lambda3, lr, lr_gamma, lr_update_gap, staggerTrainWin,
                                            stoppingThresh, trainVerboseLvl)
            Gest.data[predictedNode, :] = torch.norm(model.lin_xr2phi.weight.data[:, :n], p=2, dim=0)
            # print("Elapsed time (1) = % s seconds" % (time.time() - start))

    else:
        print("Unsupported model encountered")

    print(Gref)
    print(Gest)

    if (jobLogFilename != ""):
        if (model_name == 'eSRU_1LF' or model_name == 'eSRU_2LF'):
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
                     dim_iid_stats=dim_iid_stats,
                     dim_rec_stats=dim_rec_stats,
                     dim_final_stats=dim_final_stats,
                     dim_rec_stats_feedback=dim_rec_stats_feedback)

        else:
            np.savez(jobLogFilename, Gref=Gref, Gest=Gest.detach().cpu().numpy(), model=model_name, dataset=dataset,
                     dsid=dataset_id, T=T, F=F, nepochs=max_iter, mu1=mu1, mu2=mu2, lr=lr)

# sleep for one seconds followed by printing
# the exit key for tmux consumption
time.sleep(1)
print("#RUN_COMPLETE #RUN_COMPLETE #RUN_COMPLETE #RUN_COMPLETE")
