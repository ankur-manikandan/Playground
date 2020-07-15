import numpy as np


def second_larges_nbr(arr):
    """Function to determine the second largest 
    number given an unordered array."""

    n = len(arr)

    m1 = 0
    m2 = 0

    for i in range(n):
        if m1 < arr[i]:
            m1 = arr[i]
            m2 = m1
        if m1 < m2:
            m1 = m2
        
    print("Second largest number in array: ", m2)

if __name__ == '__main__':

    # Randomly generate data
    n = 50  # number of samples to generate
    arr = list(range(n))
    arr = np.random.choice(arr, replace=False, size=n)

    # Call function
    second_larges_nbr(arr)


"""SQL Query: Determine the employee with the second largest salary.
SELECT id, 
FROM (SELECT id, RANK() OVER(ORDER BY salary DESC) AS rank_num 
      FROM employee)
where rank_num = 2;
""" 

