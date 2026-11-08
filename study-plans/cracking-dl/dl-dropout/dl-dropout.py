import numpy as np

def dropout(X: list, mask: list, drop_prob: float, mode: str) -> list:
    """
    Returns the train-mode dropout result or unchanged test input.
    """

    if mode == 'train':
       return  ( np.array(X) * np.array(mask) ) / (1- drop_prob)
    else: 
        return X
