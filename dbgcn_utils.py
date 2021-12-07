import os
import zipfile
import numpy as np
import torch


def load_data(datapath, adjdatapath):
    A = np.load(adjdatapath)
    #X = np.load("data/node_values.npy").transpose((1, 2, 0))
    X = np.load(datapath)
    X = X.astype(np.float32)
    return A, X


def get_normalized_adj(A,flag=False):
    """
    Returns the degree normalized adjacency matrix.
    """
    #A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    if flag:
        return A
    else:
        D = np.array(np.sum(A, axis=1)).reshape((-1,))
        D[D <= 10e-5] = 10e-5    # Prevent infs
        diag = np.reciprocal(np.sqrt(D))
        A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),diag.reshape((1, -1)))
        return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(X[:, :, i: i + num_timesteps_input].transpose((0, 2, 1)))
        #features.append(X[:, 0, i: i + num_timesteps_input])
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), torch.from_numpy(np.array(target))
