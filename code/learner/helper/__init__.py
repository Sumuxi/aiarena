from .arch_config import *
# from models.net_mlp_v1 import NetworkModel
from models.arch5 import NetworkModel as net_arch5


model_dict = {
    # 'arch1_v1': NetworkModel(v1),
    # 'arch1_v2': NetworkModel(v2),
    # 'arch1_v3': NetworkModel(v3),
    'net_arch5': net_arch5(),
}