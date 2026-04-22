import pandas as pd
import numpy as np

# Sans decay: ~52% (diverge)
# Avec decay: ~67-70%
# Avec decay + normalisation: 83%

def perceptron(x, y, b, w_init, eta, epoch, decay=False):
    # labels de 0/1 en -1/1
    y = 2 * y - 1
    
    # add une col de 1
    n = x.shape[0]
    big_X = np.column_stack([np.ones(n), x.to_numpy()])
    
    w = np.array(w_init) # inti les poids
    
    for k in range(1, epoch + 1):
        # check si decay
        if decay:
            learning_rate = eta / k
        else:
            learning_rate = eta
        
        predictions = big_X @ w # calc des scores
        
        # Trouver les exemples qui violent la marge
        violations = (y * predictions) <= b # margin violation: y_i * (w^T x_i) <= b -> slides 16/26 cm3
        
        if not violations.any():
            break  # on peut exit si on a plus de margin violations (convergance)
        
        # maj du poids
        update = np.sum(y[violations, np.newaxis] * big_X[violations], axis=0)
        w = w + learning_rate * update
    
    return w


# def perceptron(x, y, b, w_init, eta, epoch, decay=False):
#     # labels de 0/1 en -1/1
#     y = 2 * y - 1
    
#     # add une col de 1 (augmentation)
#     n = x.shape[0]
#     big_X = np.column_stack([np.ones(n), x.to_numpy()])
    
#     w = np.array(w_init, dtype=float) # init les poids
    
#     for k in range(1, epoch + 1):
#         # check si decay
#         if decay:
#             eta_k = eta / k
#         else:
#             eta_k = eta
        
#         # on sélectione
#         i = (k - 1) % n  # on fait -1 pr commencer à idx 0
#         x_i = big_X[i]
#         y_i = y[i]
        
#         prediction = np.dot(w, x_i)
        
#         # si violation, on update le poids -> cf slides 16/26 cm3
#         if y_i * prediction <= b:
#             w = w + eta_k * y_i * x_i
            
#     return w

