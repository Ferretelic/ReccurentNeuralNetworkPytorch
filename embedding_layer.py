import torch
import numpy as np

class EmbeddingOnehot(torch.nn.Module):
  def __init__(self, word_size, vector_size):
    super().__init__()
    self.weights = torch.nn.Parameter(torch.randn(word_size, vector_size, dtype=torch.float32))

  def forward(self, input):
    input = torch.nn.functional.one_hot(input).float()
    output = torch.mm(input, self.weights)

    return output

class Embedding(torch.nn.Module):
  def __init__(self, word_size, vector_size):
    super().__init__()
    self.weights = torch.nn.Parameter(torch.randn(word_size, vector_size, dtype=torch.float32))

  def forward(self, input):
    output = self.weights[input]

    return output
