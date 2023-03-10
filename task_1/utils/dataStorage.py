from torch.utils.data import Dataset
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split

class dataStorage(Dataset):
    def __init__(self, path):
        if isinstance(path, Path):
            super().__init__()
            graphs = list(path.glob("*.pt"))
            train_data, test_data = train_test_split(graphs, test_size=0.2, random_state=17)
            self.train_graphs = train_data
            self.test_graphs = test_data
        else:
            self.graphs = path
    
    def append(self, path: Path):
        graphs = list(path.glob("*.pt"))
        train_data, test_data = train_test_split(graphs, test_size=0.2, random_state=17)
        self.train_graphs = self.train_graphs + train_data
        self.test_graphs = self.test_graphs + test_data

    def __getitem__(self, idx):
        return torch.load(self.train_graphs[idx])
    
    def test_get(self, idx):
        return torch.load(self.test_graphs[idx])
    
    def __len__(self) -> int:
        return len(self.train_graphs)
    
    def test_len(self):
        return len(self.test_graphs)