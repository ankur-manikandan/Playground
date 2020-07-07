import numpy as np

def determine_rem_quot(dividend, divisor):
    """Function used to determine the remainder and quotient.
    """

    r = np.inf
    quot = 0
    
    pos_dividend = abs(dividend)
    
    while r > divisor:
        
        if divisor == 0:
            print("Not defined")
            break
    
        r = pos_dividend - divisor
        pos_dividend = r
        quot += 1
        
    if dividend < 0: # or divisor < 0:
        quot = -quot
        r = -r
    
    if divisor != 0:
        print(quot, r)
    
if __name__ == '__main__':
    
    dividend = -16
    divisor = 3
    
    determine_rem_quot(dividend, divisor)