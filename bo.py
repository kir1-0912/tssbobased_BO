import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trans_sim import Rect_Opt_Flow
from tssbo import tssbo_solver
import pickle
import torch
import os

bound_dict={
    "FWn":              [10,300],   
    "Fwp":              [10,300],
    "Fn_p":             [1,50],
    "Lp":               [3,20],
    "Mul_p":            [1,20],
    "Fn_n":             [1,50],
    "Ln":               [3,20],
    "Mul_n":            [1,20],
    "Cp":               [1,10000],
    "Cs":               [1,10000],
    "indp":             [1,6],
    "inds":             [1,6],
    "K":                [0.1,0.85]
}
bound = np.array(
                [
                      [
                        bound_dict["FWn"][0],bound_dict["Fwp"][0],bound_dict["Fn_p"][0],bound_dict["Lp"][0], bound_dict["Mul_p"][0], bound_dict["Fn_n"][0],
                        bound_dict["Ln"][0],bound_dict["Mul_n"][0],bound_dict["Cp"][0],bound_dict["Cs"][0], bound_dict["indp"][0],bound_dict["inds"][0],
                        bound_dict["K"][0]
                      ]
                  ,
                    [
                        bound_dict["FWn"][1], bound_dict["Fwp"][1], bound_dict["Fn_p"][1], bound_dict["Lp"][1], bound_dict["Mul_p"][1],
                        bound_dict["Fn_n"][1],bound_dict["Ln"][1], bound_dict["Mul_n"][1], bound_dict["Cp"][1], bound_dict["Cs"][1],
                        bound_dict["indp"][1],bound_dict["inds"][1],bound_dict["K"][1]
                    ]
                  ]
                 )


def Object(x_norm):
    x_list = [item for sublist in x_norm for item in sublist]
    return Rect_Opt_Flow(x_list)


tssbo_solver(
    funct=Object,
    dim=bound.shape[1],
    bounds=bound.T,
    init_x=None,
    init_y=None,
    sigma=0.1,
    mu=0.5,
    c1=None,
    c2=None,
    allround_flag=False,
    greedy_flag=True,
    n_training=None,
    batch_size=10,
    n_candidates=200,
    n_resample=10,
    nMax=2000,
    k=13,
    dataset_file='tssbo.pkl',
    use_TS=False
)

# Load and process results
with open('tssbo.pkl', 'rb') as f:
    results = pickle.load(f)

x_samples = torch.stack(results['x'])
y_samples = torch.tensor(results['y'])



best_index = torch.argmin(y_samples)
best_x = x_samples[best_index]
best_y = y_samples[best_index]

print("Best fom:", best_y.item())
print("Best para:", best_x.tolist())
