#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split

import os

import numpy as np

import altair as alt
from vega_datasets import data


# In[6]:


class build_mlp(Model):
    
    def __init__(self, n_layers, size, size_output, activation = tf.tanh, output_activation=None):
        
        super(build_mlp,self).__init__()
        self.layers_mlp = [Dense(size,activation=activation) for _ in range(n_layers-1)]
        self.outputLayer = Dense(size_output, activation=output_activation)
        
    def call(self,input_data):
        
        x=input_data
        for layer in self.layers_mlp:
            
            x = layer(x)
            
        return self.outputLayer(x)


# In[7]:


def lrelu(x, leak=0.2):
    
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

