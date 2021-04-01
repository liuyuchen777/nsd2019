import numpy as np

import sys

sys.path.append("..")

def softmax(X):
	exps = np.exp(X - np.max(X)) 
    return exps / np.sum(exps)

def MSELoss(pre_y, target_y):
	return ((pre_y - target_y) ** 2).mean()

def MAELoss(pre_y, target_y):
	return (pre_y - target_y).abs().mean()

def L1Loss(pre_y, target_y):
	return (pre_y - target_y).abs().sum()

def L2Loss(pre_y, target_y):
	return (pre_y - target_y).square().sqrt().sum()

def cross_entropy_Loss(pre_y, target_y):
	return -1 * np.log(np.sum(targets * softmax(pre_y, -1), -1))
	
def HuberLoss(pre_y, target_y, delta = 1):
	diff = pre_y - target_y
	diff_abs = diff.abs()
	out1 = (diff ** 2) / 2
	out2 = (diff_abs - delta/2)*delta
	return np.where(diff_abs <= delta, out1, out2)
#def FocalLoss(pre_y, target_y):