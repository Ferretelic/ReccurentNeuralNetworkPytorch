import torch
import numpy as np

class RNN(torch.nn.Module):
  def __init__(self, input_vector, hidden_vector):
    super().__init__()

    self.weights_x = torch.nn.Parameter(torch.randn((input_vector, hidden_vector)))
    self.weights_h = torch.nn.Parameter(torch.randn((hidden_vector, hidden_vector)))
    self.biases = torch.nn.Parameter(torch.randn((hidden_vector,)))

  def forward(self, input_x, input_h):
    input_x = torch.mm(input_x, self.weights_x)
    input_h = torch.mm(input_h, self.weights_h)
    output = torch.add(input_x, input_h)
    output = torch.add(output, self.biases)

    return output

class TimeRNN(torch.nn.Module):
  def __init__(self, input_vector, hidden_vector, times):
    super().__init__()
    self.layers = []
    self.hs =
    for t in range(times):
      rnn = RNN(input_vector, hidden_vector)
      self.layers.append(rnn)

  def forward(self, xs, h):

    for index, layer in enumerate(self.layers):
      h = layer.forward(xs[:, index, :], h)
