from itertools import combinations
import numpy as np


def return_inds(arr, target):
    """Given an array and target, find the two numbers in the array that when 
    summed is equal to the target.
    
    Parameters
    ----------
    arr : list
        A list of numbers

    target : int
        Target value. 
        
    Returns
    -------
    ind_1 : int
        Index of the first integer in the array that contributes to the target value.
        
    ind_2 : int
        Index of the second integer in the array that contributes to the target value.
    """

    # Convert list to numpy array
    arr = np.array(arr)
    # Determine all possible combinations, excluding combinations of the same number
    arr_combs = list(combinations(arr, 2))
    
    # Determine the sum of each combination
    sum_arr = np.array(list((map(sum, arr_combs))))  
    
    # Determine the index where the sum is equal to our target
    vals = arr_combs[np.where(sum_arr == target)[0][0]]
    
    # Determine the two indices
    ind_1 = np.where(arr == vals[0])[0][0]
    ind_2 = np.where(arr == vals[1])[0][0]

    return ind_1, ind_2


if __name__ == "__main__":
    
    nums = [2, 7, 5, 1, 15]
    target = 9
    
    ind_1, ind_2 = return_inds(nums, target)
    print("Index of the first number is {} and index of the second number is {}".format(ind_1, ind_2))
