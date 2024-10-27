import os
import pyro
import torch
import pickle
import time

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pyro.distributions as dist

from tqdm.auto import trange
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from pyro.distributions import constraints
from pyro.infer.autoguide import AutoDiagonalNormal

from src.dgp_rff.deep_layer import DeepGP, DeepGPNoBias, DeepEnsembleGP
def get_num_layer(mymodel):
    return len(mymodel.model.layers)

def getvalue1(
        mymodel,
        ensemble = False
):
    #this model has num_layer layers, every layer is an ensemble layer.
    #In each ensemble layer, get omegas using layers_kernelname[0].omega
    #get weight and bias using layer.weight/bias.squeeze()
    num_layer = len(mymodel.model.layers)
    dic = {}
    if ensemble:
        for i in range(num_layer-1):
            print(i,"th (ensemble) layer:")
            print(i,"th RBF Omega = ", mymodel.model.layers[i].layers_RBF[0].layer.weight.T)#Omega
            print(i,"th RBF Weight = ", mymodel.model.layers[i].layers_RBF[1].layer.weight.T)#Weight
            print(i,"th Cauchy Omega = ", mymodel.model.layers[i].layers_Cauchy[0].layer.weight.T)#Omega
            print(i,"th Cauchy Weight = ", mymodel.model.layers[i].layers_Cauchy[1].layer.weight.T)#Weight
        print(num_layer-1, "Final Layer Weight = ", mymodel.model.layers[-1].layer.weight.T)
    else:
        for i in range(num_layer):
            # print(i,"th layer:")
            # print(i,"th Omega = ", mymodel.model.layers[i].layers[0].layer.weight.T)#Omega
            dic[str(i)+"Omega"] = mymodel.model.layers[i].layers[0].layer.weight.T
            # print(i,"th Weight = ", mymodel.model.layers[i].layers[1].layer.weight.T)#Weight
            dic[str(i)+"Weight"] = mymodel.model.layers[i].layers[1].layer.weight.T
    return dic

    #print("Cauchy Omega = ", model.model.layers[2*i].layers_Cauchy.omega)
    #print("Laplacian Omega = ", model.model.layers[2*i].layers_Laplacian.omega)
    #print("W = ", model.model.layers[2*i+1].layer.weight.squeeze())
    #print("bias = ", model.model.layers[2*i+1].layer.bias.squeeze())
def getvalue2(
        mymodel,
        ensemble = False
):
    #this model has num_layer layers, every layer is an ensemble layer.
    #In each ensemble layer, get omegas using layers_kernelname[0].omega
    #get weight and bias using layer.weight/bias.squeeze()
    num_layer = len(mymodel.model.layers)
    dic = {}
    if ensemble:
        for i in range(num_layer-1):
            print(i,"th (ensemble) layer:")
            print(i,"th RBF Omega = ", mymodel.model.layers[i].layers_RBF[0].layer.weight.T)#Omega
            print(i,"th RBF Weight = ", mymodel.model.layers[i].layers_RBF[1].layer.weight.T)#Weight
            print(i,"th Cauchy Omega = ", mymodel.model.layers[i].layers_Cauchy[0].layer.weight.T)#Omega
            print(i,"th Cauchy Weight = ", mymodel.model.layers[i].layers_Cauchy[1].layer.weight.T)#Weight
        print(num_layer-1, "Final Layer Weight = ", mymodel.model.layers[-1].layer.weight.T)
    else:
        for i in range(num_layer):
            # print(i,"th layer:")
            # print(i,"th Omega = ", mymodel.model.layers[i].layers[0].layer.weight.T)#Omega
            dic[str(i)+"Weight"] = mymodel.model.layers[i].layers[0].layer.weight.T
            # print(i,"th Weight = ", mymodel.model.layers[i].layers[1].layer.weight.T)#Weight
            # dic[str(i)+"Weight"] = mymodel.model.layers[i].layers[1].layer.weight.T
    return dic