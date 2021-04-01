'''
Author: Liu Yuchen
Date: 2021-04-02 19:29:53
LastEditors: Liu Yuchen
LastEditTime: 2021-04-02 07:42:41
Description: 
FilePath: /Local_Lab/nsd_2019/term_project/simpleCNN/Layers/Linear.py
GitHub: https://github.com/liuyuchen777
'''
import numpy as np

class Linear():
    def __init__(self, input_dim, output_dim, bias = False):
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.is_bias    = bias
        self.weight     = np.random.uniform(-np.sqrt(input_dim),np.sqrt(input_dim),(input_dim, output_dim))
        self.bias       = np.random.uniform(-np.sqrt(input_dim),np.sqrt(input_dim),(output_dim))
        self.reg        = np.zeros((output_dim), dtype = np.float16) # for backpropagation
    
    def __str__(self):
        return "Linear(input_dim = %d, output_dim = %d, bias = %r)" %(self.input_dim,self.output_dim,self.is_bias)
    
    def forward(self, input_data):
        x = np.asmatrix(input_data)
        self.reg = x
        x = x * np.asmatrix(self.weight)
        x = np.asarray(x)
        if self.is_bias:
            x += self.bias
        return x
    
    def backprop(self, delta_loss, lr = 0.1):
        delta_w = np.asmatrix(self.reg).T * np.asmatrix(delta_loss)
        self.weight -= lr*delta_w
        if self.is_bias:
            self.bias -= lr*np.mean(np.asarray(delta_loss), axis = 0)
        return delta_loss * (self.weight+lr*delta_w).T