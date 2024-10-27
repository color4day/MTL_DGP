# write a teacher layer for test
# model2: teacher layer
# model3: not add xtrain in teacher layer
# model4: another student layer
import os
import pyro
import torch
import pickle

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pyro.distributions as dist

from torch import Tensor
from tqdm.auto import trange
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from pyro.distributions import constraints
from pyro.infer.autoguide import AutoDiagonalNormal

# os.chdir("../../")

# from src.dgp_rff.outer_layer import SingleGP, SingleCauchyGP
from src.dgp_rff.deep_layer import DeepGP, DeepGPNoBias, DeepEnsembleGP



import torch
torch.cuda.is_available()
torch.cuda.current_device()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


class ModelDGP(PyroModule):
    def __init__(self, dim_list=[1, 1, 1], J_list=[50, 10]):
        super().__init__()

        self.out_dim = dim_list[-1]
        self.model = DeepGP(dim_list, J_list)
        self.model.to('cuda')

    def forward(self, x, y=None):
        mu = self.model(x).squeeze()  # 10000*6

        scale = pyro.sample("sigma",
                            dist.Gamma(torch.tensor(0.5, device='cuda'), torch.tensor(1.0, device='cuda'))).expand(
            self.out_dim)  # Infer the response noise

        # Sampling model
        with pyro.plate("data", x.shape[0]):  # x.shape[0]=10000
            # obs = xxx("obs", mu, obs=y)
            obs = pyro.sample("obs", dist.MultivariateNormal(mu.cuda(), torch.diag(scale * scale).cuda()), obs=y)

        return mu

def buildmodel(modeltoken, x_train, y_train):
    if modeltoken == 0:
        model = ModelDGP(dim_list=[x_train.shape[1], 10, y_train.shape[1]], J_list=[50, 50])
    if modeltoken == 1:
        model = ModelDGP(dim_list=[x_train.shape[1], 10, y_train.shape[1]], J_list=[50, 50])
    if modeltoken == 2:
        model = ModelDGP(dim_list=[x_train.shape[1], 10, y_train.shape[1]], J_list=[50, 50])
    if modeltoken == 3:
        model = ModelDGP(dim_list=[x_train.shape[1], 10, y_train.shape[1]], J_list=[50, 50])
    if modeltoken == 4:
        model = ModelDGP(dim_list=[x_train.shape[1], 10, y_train.shape[1]], J_list=[50, 50])
    print(x_train.shape)
    print(y_train.shape)
    model = model.to('cuda')
    return model

def trainmodel(model1, x_train, y_train,iter = 15000):
    mean_field_guide = AutoDiagonalNormal(model1)
    optimizer = pyro.optim.Adam({"lr": 0.001})

    svi = SVI(model1, mean_field_guide, optimizer, loss=Trace_ELBO())
    pyro.clear_param_store()

    num_epochs = iter
    progress_bar = trange(num_epochs)
    l = 500
    losslist1 = []

    interval = max(num_epochs // l, 1)

    for epoch in progress_bar:
        loss = svi.step(x_train, y_train)
        if epoch % interval == 0:
            losslist1.append(loss / x_train.shape[0])
        progress_bar.set_postfix(loss=f"{loss / x_train.shape[0]:.3f}")

    return model1, mean_field_guide, losslist1

def predmodel(model1, mean_field_guide, x_test, x_obs, y_obs):
    predictive1 = Predictive(model1, guide=mean_field_guide, num_samples=500)
    preds1 = predictive1(x_test)
    y_pred1 = preds1['obs'].cpu().detach().numpy().mean(axis=0)
    y_dim = y_obs.shape[1]
    for d in range(y_dim):
        plt.plot(x_obs, y_obs[:, d], label="Observation")
        plt.plot(x_test.cpu(), y_pred1[:, d], label="Prediction")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # Set random seed for reproducibility
    np.random.seed(42)

    # Read data
    cwd = os.getcwd()
    print(cwd)
    index = "dgp"
    fold = 4

    X_train_path = os.path.join(cwd, "folds", "synthetic_"+str(index)+"_fold_"+str(fold)+"_X_train.txt")
    X_test_path = os.path.join(cwd, "folds", "synthetic_"+str(index)+"_fold_"+str(fold)+"_X_test.txt")
    Y_train_path = os.path.join(cwd, "folds", "synthetic_"+str(index)+"_fold_"+str(fold)+"_Y_train.txt")
    Y_test_path = os.path.join(cwd, "folds", "synthetic_"+str(index)+"_fold_"+str(fold)+"_Y_test.txt")

    x_obs = np.loadtxt(X_train_path)
    y_obs = np.loadtxt(Y_train_path)
    x_val = np.loadtxt(X_test_path)
    y_val = np.loadtxt(Y_test_path)

    # Set plot limits and labels
    xlims = [-0.2, 0.2]

    # The X and Y have to be at least 2-dim
    x_train = torch.from_numpy(x_obs).float().reshape(-1,1)
    y_train = torch.from_numpy(y_obs).float()
    x_test = torch.from_numpy(x_val).float().reshape(-1,1)
    y_test = torch.from_numpy(y_val).float()

    x_train = x_train.cuda()
    y_train = y_train.cuda()
    x_test = x_test.cuda()
    y_test = y_test.cuda()

