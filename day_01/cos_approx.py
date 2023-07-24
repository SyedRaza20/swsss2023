#!/usr/bin/env python
"""Space 477: Python: I

cosine approximation function
"""
__author__ = 'Syed Raza'
__email__ = 'sar0033@uah.edu'

from math import factorial
from math import pi


def cos_approx(x, accuracy=10):
    """
    What it does? This function returns the approximation of the cosine of x
    based on Taylor's expansion until the co-effecient accuracy
    
    Args:
        x (float): 
            x is the value that we approximate the cosine of 
        accuracy (int):
            the accuracy is the Taylor coefficient until which we approximate
            cosine of x
    
    Returns:
        This function returns one value (cos_approx_value) that is the cosine
        of x
    """
    # initializing the return value below:
    cos_approx_value = 0
    
    # for loop for the Taylor's expansion:
    for n in range(accuracy+1):
        # make the formula:
        cos_approx_value = cos_approx_value + ((-1)**n/(factorial(2*n)))*(x**(2*n))
    
    return cos_approx_value

# Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    print("cos(0) = ", cos_approx(0))
    print("cos(pi) = ", cos_approx(pi))
    print("cos(2*pi) = ", cos_approx(2*pi))
    print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))
    
    # Writing an assert statement for this function:
    assert cos_approx(0) > 1-1.e-2 and cos_approx(0) < 1+1.e-2, "cos(0) is not even 1"
    assert cos_approx(pi) > -1-1.e-2 and cos_approx(pi) < -1+1.e-2, "cos(pi) is not even 0"