import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

import torch

# List of tensors (example tensors)
tensor_list = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])]

# Convert the list of tensors to a single tensor
combined_tensor = torch.stack(tensor_list)

print(combined_tensor)
print(combined_tensor.shape)