import torch
from torch.autograd import Variable

import torch.nn.functional as F
from torch import nn

'''
Adapted from Eric Jang's TF implementation @ https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
'''
def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.distributions.Uniform(low=torch.zeros(1), high=torch.ones(1))
    samples = U.sample_n(shape)
    return -torch.log(-torch.log(samples + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape[0]).view(-1,)
    return torch.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    k = logits.size()[-1]
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)

    # this is the straight through gradient estimator?
    y = tf.stop_gradient(y_hard - y) + y

  return y

if __name__ == '__main__':

    logits = torch.tensor([1,2,3,4]).float()
    sample = gumbel_softmax_sample(logits, temperature=torch.tensor(.5)))
    print(sample)