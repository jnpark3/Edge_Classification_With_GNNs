from torch.utils.data import Dataset
from pathlib import Path
import torch

class dataStorage(Dataset):
    def __init__(self, path: Path):
        super().__init__()
        self.graphs = list(path.glob("*.pt"))
    
    def append(self, path: Path):
        self.graphs = self.graphs + list(path.glob("*.pt"))

    def __getitem__(self, idx):
        return torch.load(self.graphs[idx])
    
    def __len__(self) -> int:
        return len(self.graphs)