# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:22:20 2018

@author: Lenovo
"""

class Config:

    explore = 4000000
    n_actions = 4
    learning_rate = 0.001
    gamma = 0.9
    replace_target_iter = 2000
    memory_size = 10000
    batch_size = 256
    final_epsilon = 0.001
    initial_epsilon = 0.001
    observe = 1000
    model_file = './model/snake' 