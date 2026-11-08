import numpy as np

def activation_functions(x: float, activation: str) -> list:
    """
    Returns the activation value and analytical derivative.
    """
    
    if activation == 'relu':
        f_x = np.maximum(x, 0)
        f_x_prime = np.where(f_x > 0, 1, 0 )
        return [f_x, f_x_prime]
    elif activation == 'leaky_relu':
        f_x = np.where(x > 0, x, 0.01 * x)
        f_x_prime = np.where(x > 0, 1, 0.01)
        return [f_x, f_x_prime]
    elif activation == 'sigmoid':
        f_x = (1 / (1 + np.exp(-x)))
        f_x_prime = f_x * (1- f_x)
        return [f_x, f_x_prime]
    elif activation == 'tanh': 
        f_x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        f_x_prime = 1 - np.square(f_x)
        return [f_x, f_x_prime]
    elif activation == 'gelu':
        c=np.sqrt(2.0/np.pi); inner=c*(x+0.044715*x**3)  
        tanh_inner=np.tanh(inner)
        output=0.5*x*(1.0+tanh_inner)
        derivative=0.5*(1.0+tanh_inner)+0.5*x*(1.0-tanh_inner**2)*c*(1.0+3.0*0.044715*x**2)

        return [output, derivative]
        
    elif activation== 'swish':
         sigmoid=1.0/(1.0+np.exp(-x)); output=x*sigmoid; derivative=sigmoid+x*sigmoid*(1.0-sigmoid)
        
         return [output, derivative]
