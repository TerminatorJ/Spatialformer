import torch
from dataclasses import dataclass
import json

#coordinating the settings of the model
with open('/scratch/project_465001820/scCLIP/scCLIP/config/_config_train.json') as f:
    config = json.load(f)

@dataclass
class Settings:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: torch.dtype = torch.bfloat16
    batch_size: int = config['batch_size']
