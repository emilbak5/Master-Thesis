import numpy as np
import itertools
import math


# make a function list of string and return the total number of combinations. Here is a few rules:
# 1. A combination has to be unique.
# 2. ['a', 'b', 'c'] is the same as ['c', 'b', 'a'] and ['b', 'c', 'a'] and so on and should be counted as one.
# 3. Not all strings has be used in a combination. For example ['a', 'b', 'c'] can be ['a', 'b'] or ['a', 'c'] or ['b', 'c'] or ['a'] or ['b'] or ['c'].
# 4. a combination cannot be empty ie. [] is not a valid combination.

def get_combinations(lst):
    unique_combinations = set()
    for i in range(1, len(lst) + 1):
        for combination in itertools.combinations(lst, i):
            unique_combinations.add(tuple(sorted(combination)))
        
    unique_combinations = list(unique_combinations)
        # sort the list after the length of each tuple
    unique_combinations.sort(key=lambda x: len(x))


    return unique_combinations

def count_combinations(lst):

    n = len(lst)
    count = 0
    for k in range(1, n + 1):
        count += math.comb(n, k)
    return count


if __name__ == "__main__":
    print(get_combinations(['1', '2', '3']))
    print(get_combinations(['a', 'b', 'c', 'd', 'e', 'f', 'g']))

    print(count_combinations(['1', '2', '3']))
    print(count_combinations(['a', 'b', 'c', 'd', 'e', 'f', 'g']))
    