import torch

if torch.cuda.is_available():
    DEVICE = torch.device('metal')
else:
    DEVICE = torch.device('cpu')

print("Device:", DEVICE)
