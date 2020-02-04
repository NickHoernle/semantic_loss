#!/usr/bin/env python

import sys
from functools import lru_cache

from argparse import ArgumentParser
from collections import deque
import numpy as np
import torch
from tqdm import tqdm

'''
Generate knapsack of N items. Each item has a weight and a cost.

(a) for N items, sample the weight and the cost

(b) solve the napsack problem for the optimal solution

(c) store the input problems in "knapsack_{id}_{knapsack_limit}

(d) store the indexes (as a one hot array of length N) in solutions file:
id: {id}, id_str: knapsack_{id}_{knapsack_limit}, limit: {knapsack_limit}, solution: solution_str
'''

def create_knapsack(total_num_items):
    '''
    Create an instance of the knapsack problem with total_num_items being the size of the 
    items that we create.
    
    Aim to have 25% of the items in the knapsack solution
    
    return [(item_weight, item_price)], knapsack_limit
    '''

    knapsack_limit = np.random.choice(np.arange(50,200,1))

    rate = knapsack_limit*4/total_num_items
    solved = False

    while not solved:
        items = []
        partial_sum = 0
        for item in range(total_num_items):
            weight = 0
            price = 0
            while (weight == 0) or (price == 0):
                rate_ = np.random.gamma(rate/4, rate)
                weight = np.random.poisson(rate_)
                price = np.random.poisson(rate_)
            items.append((weight, price, item))
            partial_sum += weight

        if partial_sum > knapsack_limit:
            solved = True

    return items, knapsack_limit

def knapsack(items, maxweight):
    """Solve the knapsack problem by finding the most valuable subsequence
    of items that weighs no more than maxweight.

    items must be a sequence of pairs (value, weight), where value is a
    number and weight is a non-negative integer.

    maxweight is a non-negative integer.

    Return a pair whose first element is the sum of values in the most
    valuable subsequence, and whose second element is the subsequence.

    >>> items = [(4, 12), (2, 1), (6, 4), (1, 1), (2, 2)]
    >>> knapsack(items, 15)
    (11, [(2, 1), (6, 4), (1, 1), (2, 2)])

    """
    @lru_cache(maxsize=None)
    def bestvalue(i, j):
        # Return the value of the most valuable subsequence of the first
        # i elements in items whose weights sum to no more than j.
        if j < 0:
            return float('-inf')
        if i == 0:
            return 0
        value, weight, id = items[i - 1]
        return max(bestvalue(i - 1, j), bestvalue(i - 1, j - weight) + value)

    j = maxweight
    result = []
    for i in reversed(range(len(items))):
        if bestvalue(i + 1, j) != bestvalue(i, j):
            result.append(items[i])
            j -= items[i][1]
    result.reverse()
    return bestvalue(len(items), maxweight), result


def main():

    import os
    import torch
    import shutil
    import json

    path = '../data/'
    shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    meta_data = {}
    np.random.seed(11)

    for i in tqdm(range(10000)):

        items, maxweight = create_knapsack(100)
        value, solution_seq = knapsack(items, maxweight)

        items = torch.tensor(items)
        solution_seq = torch.tensor(solution_seq)

        with open(os.path.join(path, f'items_{i}.pt'), 'wb') as f:
            torch.save(items, f)

        with open(os.path.join(path, f'solutions_{i}.pt'), 'wb') as f:
            torch.save(solution_seq, f)

        meta_data[i] = {
            'max_weight': int(maxweight),
            'best_value': int(value)
        }

    with open(os.path.join(path, f'meta_data.json'), 'w') as f:
        json.dump(meta_data, f)


if __name__ == '__main__':
   sys.exit(main())
