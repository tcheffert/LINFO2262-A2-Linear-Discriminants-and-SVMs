import pandas as pd
import numpy as np

def predict(w, x):
    """
    Predicts the class labels of the data matrix x
    Inputs:    w a vector with the weight values defining a linear discriminant (intercept in w[0]) [list or numpy array of size d+1]
               x a feature matrix containing an example on each row [pandas DataFrame of shape n x d]
    Output:    A vector of labels (1 or 0) [numpy array of size n]
    """
    n = x.shape[0]
    big_X = np.column_stack([np.ones(n), x.to_numpy()]) # la collonne de 1 comme à la Q1
    
    # w^T * x
    scores = big_X @ w
    
    # cast labels en 0/1
    for i in range(len(scores)):
        if scores[i] > 0:
            scores[i] = 1
        else:
            scores[i] = 0
    
    return scores
   
