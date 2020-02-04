from torch.utils import data
import os
import json
import numpy as np
import torch
from tqdm import tqdm

class KnapsackLoader(data.Dataset):

    def __init__(self, data_path='../data', dset_type='train'):

        with open(os.path.join(data_path, 'meta_data.json'), 'r') as f:
            self.metadata = json.load(f)

        self.solutions = self.metadata
        self.data_path = data_path

        indexes = {'train': np.arange(0,7000).astype(int),
                   'test': np.arange(7000,8500).astype(int),
                   'valid': np.arange(8500,10000).astype(int)}[dset_type]

        self.list_IDs = np.fromiter(self.metadata.keys(), dtype=int)[indexes]
        print(self.list_IDs)
        # print(np.array(self.metadata.keys()))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        items = torch.load(os.path.join(self.data_path, f'items_{ID}.pt'))
        items = items[:,(0,1)].view(-1,)
        solutions = torch.zeros(100)
        these_sol = torch.load(os.path.join(self.data_path, f'solutions_{ID}.pt'))[:,-1].view(-1,)
        solutions[these_sol] = 1

        return items, solutions,\
               torch.tensor(self.metadata[f'{ID}']['max_weight']),\
               torch.tensor(self.metadata[f'{ID}']['best_value'])

if __name__ == '__main__':
    params = {'batch_size': 512,
              'shuffle': True,
              'num_workers': 6}

    training_set = KnapsackLoader(data_path='../data', dset_type='train')
    training_generator = data.DataLoader(training_set, **params)

    for local_batch, local_labels, max_weights, best_values in training_generator:
        # print(local_labels.sum(dim=0))
        pass

    print('Test loading complete')