import numpy as np

def lms(x, y, b, w_init, eta, epoch, decay=True):
    # idem base que pr la Q1
    y = 2 * y - 1
    
    n = x.shape[0]
    big_X = np.column_stack([np.ones(n), x.to_numpy()])
    
    w = np.array(w_init)
    
    for k in range(1, epoch + 1):
        if decay:
            eta_k = eta / k
        else:
            eta_k = eta
        
        # cf slide 19/26 cm3
        i = (k - 1) % n
        x_i = big_X[i]
        y_i = y[i]
        
        # maj Widrow-Hoff
        # cf slide: w(k+1) = w(k) + eta(k) * (b_k - w^T * x_k) * x_k
        prediction = np.dot(w, x_i)
        target = y_i * b   # cible b_k est (yi * b)
        
        error = target - prediction
        
        # maj vecteur de poids
        w = w + eta_k * error * x_i

        # normalisation
        norm = np.linalg.norm(w)
        if norm > 1:
            w = w / norm
        
    return w