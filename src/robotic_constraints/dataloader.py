import os
import torch
import json
from torch.utils import data

class NavigateFromTo(data.Dataset):

  def __init__(self, type='train', data_path='../../data/robotic_constraints/', trajectory=False):
        self.data_path = data_path
        # load the ids list
        file_path = os.path.join(data_path, 'data_assignments.json')
        with open(file_path, 'r') as f:
            self.ids_all = json.load(f)

        self.list_IDs = self.ids_all[type]
        # TODO implement a subsampler to limit data
        # self.trajectory = trajectory

  @property
  def n_dims(self):
      params = self.__getitem__(0)
      return params[0].size()[0]

  @property
  def constraints(self):
      return self.ids_all['constraints']

  def __len__(self):
        return len(self.list_IDs)

  def __getitem__(self, index):
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        weight_path = os.path.join(self.data_path, f"weights_{ID}.pt")
        trajectory_path = os.path.join(self.data_path, f"trajectory_{ID}.pt")

        weights = torch.load(weight_path)
        trajectory_dat = torch.load(trajectory_path)
        condition_params = torch.tensor([trajectory_dat[0,0], trajectory_dat[0,1], trajectory_dat[-1,0], trajectory_dat[-1,1]])

        return weights.view(-1,), condition_params, trajectory_dat.view(-1,)