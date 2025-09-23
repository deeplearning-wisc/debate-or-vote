from datasets import load_dataset, Dataset, concatenate_datasets
import random, os, pickle
import numpy as np
import pandas as pd
from copy import deepcopy

from data.data_utils import *


def load_data(args, split=None, easy=False):
    
    data_size = args.data_size
    num_params = 4 if easy else 6

    if split == 'train' :
        x = np.random.default_rng(0).integers(0, 30, size=num_params * data_size)
    else :
        x = np.random.default_rng(1).integers(0, 30, size=num_params * data_size)

    X, Y = [], []
    for i in range(0, num_params * data_size, num_params):
        if easy :
            a, b, c, d = x[i:i+4]
            question = f'What is the result of {a}+{b}*{c}-{d}?'
            answer = a + b * c - d
        else :
            a, b, c, d, e, f = x[i:i+6]
            if f == 0 : 
                f = 1
            question = f'What is the result of {a}+{b}*{c}+{d}-{e}÷{f}?'
            answer = a + b * c + d - e / f
        X.append(question)
        Y.append(answer)
    
    return X, Y