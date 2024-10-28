import torch

from networkmodel.pytorch.NetworkModel import NetworkModel

model = NetworkModel()
state_dict = torch.load("../assets/baseline/model.pth", map_location="cpu")
missing_keys, unexpected_keys = model.load_state_dict(state_dict["network_state_dict"], strict=True)
print(f"missing_keys: {missing_keys}")
print(f"unexpected_keys: {unexpected_keys}")
